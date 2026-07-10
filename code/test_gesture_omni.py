from __future__ import annotations

import argparse
import json
import re
import shutil
import traceback
from pathlib import Path

import torch
from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_ID = "Qwen/Qwen2.5-Omni-7B"
DEFAULT_VIDEO = PROJECT_ROOT / "materials" / "gestures" / "Homogeneity1_TargetPremise(Positive).mp4"
DEFAULT_OUTPUT = PROJECT_ROOT / "results" / "qwen__qwen2.5-omni-7b_gesture_ratings.jsonl"

STATEMENTS = {
    "target": "Sam will take all of the coins.",
    "baseline": "Sam will take some, but not all, of the coins.",
}


def model_slug(model_id: str) -> str:
    slug = model_id.lower().replace("/", "__")
    slug = re.sub(r"[^a-z0-9_.-]+", "_", slug)
    return slug.strip("_")


def rating_prompt(statement: str) -> str:
    return (
        "To what degree do the video and audio together suggest the following statement?\n\n"
        f"\"{statement}\"\n\n"
        "Answer with a single integer from 0 to 100.\n"
        "0 = not suggested at all.\n"
        "100 = very strongly suggested.\n"
        "Output the number and why."
    )


def build_conversation(
    video_path: Path,
    statement: str,
) -> list[dict]:
    return [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": (
                        "You are a multimodal assistant. Answer using only the provided video, "
                        "including visible gestures and spoken content."
                    ),
                }
            ],
        },
        {
            "role": "user",
            "content": [
                {"type": "video", "video": str(video_path.resolve())},
                {"type": "text", "text": rating_prompt(statement)},
            ],
        },
    ]


def run_one(
    model: Qwen2_5OmniForConditionalGeneration,
    processor: Qwen2_5OmniProcessor,
    video_path: Path,
    statement: str,
    max_new_tokens: int,
    use_audio_in_video: bool,
) -> tuple[str, list[dict]]:
    try:
        from qwen_omni_utils import process_mm_info
    except ImportError as exc:
        raise ImportError(
            "Qwen2.5-Omni video loading requires qwen-omni-utils. "
            "Install it with: pip install 'qwen-omni-utils[decord]' -U"
        ) from exc

    conversation = build_conversation(
        video_path=video_path,
        statement=statement,
    )
    text = processor.apply_chat_template(
        conversation,
        add_generation_prompt=True,
        tokenize=False,
    )
    audios, images, videos = process_mm_info(
        conversation,
        use_audio_in_video=use_audio_in_video,
    )
    inputs = processor(
        text=text,
        audio=audios,
        images=images,
        videos=videos,
        return_tensors="pt",
        padding=True,
        use_audio_in_video=use_audio_in_video,
    )
    inputs = inputs.to(model.device).to(model.dtype)

    with torch.inference_mode():
        text_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            return_audio=False,
            use_audio_in_video=use_audio_in_video,
            do_sample=False,
        )

    response = processor.batch_decode(
        text_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0].strip()
    return response, conversation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pilot Qwen2.5-Omni on one gesture video inference item."
    )
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--video", type=Path, default=DEFAULT_VIDEO)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--statement-types",
        nargs="+",
        choices=tuple(STATEMENTS),
        default=list(STATEMENTS),
    )
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument(
        "--no-audio-in-video",
        action="store_true",
        help="Ignore the video's audio track. By default, audio is used because the premise is spoken.",
    )
    parser.add_argument("--flash-attention-2", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    video_path = args.video
    if not video_path.is_absolute():
        video_path = PROJECT_ROOT / video_path
    if not video_path.exists():
        raise FileNotFoundError(f"Cannot find gesture video: {video_path}")
    if not args.no_audio_in_video and not (
        shutil.which("ffmpeg") or shutil.which("avconv")
    ):
        raise RuntimeError(
            "Audio is enabled because the premise is spoken in the video, but neither "
            "ffmpeg nor avconv is available on PATH. Install/load ffmpeg on the cluster, "
            "or rerun with --no-audio-in-video to ignore the spoken audio."
        )

    jobs = [(statement_type, STATEMENTS[statement_type]) for statement_type in args.statement_types]
    print(f"Found {len(jobs)} gesture statement job(s).")
    print(f"Video: {video_path}")
    if args.validate_only:
        for statement_type, statement in jobs:
            preview = {
                "model": args.model_id,
                "source_video": str(video_path),
                "statement_type": statement_type,
                "statement": statement,
                "conversation": build_conversation(
                    video_path=video_path,
                    statement=statement,
                ),
            }
            print(json.dumps(preview, ensure_ascii=False, indent=2))
        print("Validation passed.")
        return

    load_kwargs = {
        "torch_dtype": "auto",
        "device_map": "auto",
    }
    if args.flash_attention_2:
        load_kwargs["attn_implementation"] = "flash_attention_2"

    print(f"Loading model: {args.model_id}")
    model = Qwen2_5OmniForConditionalGeneration.from_pretrained(args.model_id, **load_kwargs)
    model.disable_talker()
    processor = Qwen2_5OmniProcessor.from_pretrained(args.model_id)
    print("CUDA available:", torch.cuda.is_available())
    print("Model device:", next(model.parameters()).device)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        for statement_type, statement in jobs:
            print(f"Running: {video_path.name} [{statement_type}]")
            try:
                response, conversation = run_one(
                    model=model,
                    processor=processor,
                    video_path=video_path,
                    statement=statement,
                    max_new_tokens=args.max_new_tokens,
                    use_audio_in_video=not args.no_audio_in_video,
                )
                record = {
                    "model": args.model_id,
                    "runner": "qwen25_omni",
                    "source_video": str(video_path),
                    "statement_type": statement_type,
                    "statement": statement,
                    "response": response,
                    "use_audio_in_video": not args.no_audio_in_video,
                    "conversation": conversation,
                }
                print(f"Output: {response}")
            except Exception as exc:
                traceback_text = traceback.format_exc()
                record = {
                    "model": args.model_id,
                    "runner": "qwen25_omni",
                    "source_video": str(video_path),
                    "statement_type": statement_type,
                    "statement": statement,
                    "error": repr(exc),
                    "traceback": traceback_text,
                    "use_audio_in_video": not args.no_audio_in_video,
                }
                print(f"Error on {video_path.name} [{statement_type}]: {exc!r}")
                print(traceback_text)

            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            f.flush()
            print("-" * 80)

    print(f"Wrote results to {args.output}")


if __name__ == "__main__":
    main()
