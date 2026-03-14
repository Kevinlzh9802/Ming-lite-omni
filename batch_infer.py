#!/usr/bin/env python3
"""Batch inference over subfolders containing (.mp4, .wav) pairs.

Expected layout:
<data_root>/
  subfolder_A/
    clip_01.mp4
    clip_01.wav
  subfolder_B/
    ...

Output JSON:
{
  "subfolder_A": [
    {"file": "clip_01", "response": "..."}
  ],
  "subfolder_B": [...]
}
"""

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List

import torch
from transformers import AutoProcessor

from modeling_bailingmm import BailingMMNativeForConditionalGeneration

PROMPTS_DIR = Path("./prompts")


def natural_sort_key(value: str):
    """Sort strings by text chunks and embedded integers."""
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", value)]


def collect_av_pairs(data_root: Path) -> Dict[str, List[dict]]:
    """Collect matched .mp4/.wav pairs in each immediate child directory."""
    if not data_root.is_dir():
        raise FileNotFoundError(f"Data root does not exist: {data_root}")

    result: Dict[str, List[dict]] = {}
    for subfolder in sorted(data_root.iterdir(), key=lambda path: natural_sort_key(path.name)):
        if not subfolder.is_dir():
            continue

        pairs: List[dict] = []
        video_files = {
            f.stem: f for f in sorted(subfolder.glob("*.mp4"), key=lambda path: natural_sort_key(path.name))
        }
        for stem, video_path in sorted(video_files.items(), key=lambda item: natural_sort_key(item[0])):
            audio_path = subfolder / f"{stem}.wav"
            if not audio_path.exists():
                print(f"[WARN] Missing audio for {video_path}, skip.")
                continue
            pairs.append(
                {
                    "stem": stem,
                    "video": str(video_path),
                    "audio": str(audio_path),
                }
            )

        if pairs:
            result[subfolder.name] = pairs

    return result


def load_model_and_processor(model_path: str, device: str, attn_implementation: str):
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)

    kwargs = {
        "torch_dtype": torch.bfloat16,
        "attn_implementation": attn_implementation,
        "low_cpu_mem_usage": True,
    }
    if device == "cuda":
        kwargs["device_map"] = {"": 0}

    model = BailingMMNativeForConditionalGeneration.from_pretrained(model_path, **kwargs)
    if device != "cuda":
        model = model.to(device)
    model = model.eval()
    return model, processor


@torch.inference_mode()
def infer_messages(
    model,
    processor,
    messages: List[dict],
    max_new_tokens: int,
) -> str:
    text = processor.apply_chat_template(messages, add_generation_prompt=True)
    image_inputs, video_inputs, audio_inputs = processor.process_vision_info(messages)

    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        audios=audio_inputs,
        return_tensors="pt",
    ).to(model.device)

    for key in inputs.keys():
        if key in {"pixel_values", "pixel_values_videos", "audio_feats"}:
            inputs[key] = inputs[key].to(dtype=torch.bfloat16)

    generated_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        use_cache=False,
        eos_token_id=processor.gen_terminator,
    )
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]

    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]
    return output_text


def run_batch(
    model,
    processor,
    pairs_by_subfolder: Dict[str, List[dict]],
    first_turn_prompt: str,
    later_turn_prompt: str,
    max_new_tokens: int,
    max_frames: int,
    sample: str,
) -> Dict[str, List[dict]]:
    output: Dict[str, List[dict]] = {}

    for subfolder, pairs in pairs_by_subfolder.items():
        print(f"[INFO] Processing {subfolder}: {len(pairs)} pairs")
        items: List[dict] = []
        messages: List[dict] = []
        for pair_idx, pair in enumerate(pairs):
            stem = pair["stem"]
            prompt = first_turn_prompt if pair_idx == 0 else later_turn_prompt
            messages.append(
                {
                    "role": "HUMAN",
                    "content": [
                        {"type": "video", "video": pair["video"], "max_frames": max_frames, "sample": sample},
                        {"type": "audio", "audio": pair["audio"]},
                        {"type": "text", "text": prompt},
                    ],
                }
            )
            try:
                response = infer_messages(
                    model=model,
                    processor=processor,
                    messages=messages,
                    max_new_tokens=max_new_tokens,
                )
                messages.append(
                    {
                        "role": "ASSISTANT",
                        "content": [{"type": "text", "text": response}],
                    }
                )
                print(f"[OK] {subfolder}/{stem}")
            except Exception as exc:
                response = f"[ERROR] {type(exc).__name__}: {exc}"
                messages.pop()
                print(f"[ERR] {subfolder}/{stem}: {response}")

            items.append(
                {
                    "file": stem,
                    "response": response,
                }
            )

        output[subfolder] = items

    return output


def load_prompt(prompt_path: Path) -> str:
    if not prompt_path.is_file():
        raise FileNotFoundError(f"Prompt file not found: {prompt_path}")
    prompt = prompt_path.read_text(encoding="utf-8").strip()
    if not prompt:
        raise ValueError(f"Prompt file is empty: {prompt_path}")
    return prompt


def load_prompts(prompt: str) -> tuple[str, str]:
    first_turn_path = PROMPTS_DIR / f"{prompt}_1.txt"
    later_turn_path = PROMPTS_DIR / f"{prompt}_after.txt"
    return load_prompt(first_turn_path), load_prompt(later_turn_path)


def main():
    parser = argparse.ArgumentParser(description="Batch AV-pair inference for Ming-Lite-Omni.")
    parser.add_argument("--data-root", type=str, required=True, help="Parent folder containing subfolders with AV pairs.")
    parser.add_argument("--output-json", type=str, required=True, help="Path to write aggregated JSON output.")
    parser.add_argument("--prompt", type=str, required=True, help="Prompt prefix in ./prompts, e.g. foo -> foo_1.txt and foo_after.txt.")
    parser.add_argument("--model-path", type=str, default=".")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--attn-implementation", type=str, default="flash_attention_2")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--max-frames", type=int, default=128)
    parser.add_argument("--sample", type=str, default="uniform", choices=["uniform", "fps"])
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is requested but not available.")

    first_turn_prompt, later_turn_prompt = load_prompts(args.prompt)
    print(f"[INFO] Loaded first-turn prompt from {PROMPTS_DIR / f'{args.prompt}_1.txt'}")
    print(f"[INFO] Loaded later-turn prompt from {PROMPTS_DIR / f'{args.prompt}_after.txt'}")

    data_root = Path(args.data_root)
    pairs = collect_av_pairs(data_root)
    if not pairs:
        raise RuntimeError(f"No valid (.mp4, .wav) pairs found under {data_root}")

    model, processor = load_model_and_processor(
        model_path=args.model_path,
        device=args.device,
        attn_implementation=args.attn_implementation,
    )

    results = run_batch(
        model=model,
        processor=processor,
        pairs_by_subfolder=pairs,
        first_turn_prompt=first_turn_prompt,
        later_turn_prompt=later_turn_prompt,
        max_new_tokens=args.max_new_tokens,
        max_frames=args.max_frames,
        sample=args.sample,
    )

    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[INFO] Saved results to {output_path}")


if __name__ == "__main__":
    main()
