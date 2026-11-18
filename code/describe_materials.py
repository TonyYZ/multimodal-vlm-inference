from __future__ import annotations

import argparse
import json
import traceback
from pathlib import Path

import feed_experiment as feed


PROMPT_VERSION = "5"

# Pure video: the whole story is one video of slides. Split: the same slides are given in order as
# text and short video clips, so the wording names both.
DESCRIPTION_PROMPTS = {
    "pure-video": "Here's a video with several slides.",
    "split": "Here's a story shown as several slides, given in order as text and short video clips.",
    "pure-text": "Here's a story shown as several slides, given as text.",
}
DESCRIPTION_INSTRUCTION = (
    "Describe each slide in words. For each slide, start the sentence with "
    "Slide X (the number of the slide)."
)


def default_output_path(model_id: str, input_mode: str) -> Path:
    mode = input_mode.replace("-", "_")
    return (
        feed.PROJECT_ROOT / "results" / "animation" / "perception"
        / "descriptions"
        / f"{feed.model_slug(model_id)}_{mode}_descriptions.jsonl"
    )


def description_content(sequence: dict, input_mode: str) -> list[dict]:
    return [
        {"type": "text", "text": DESCRIPTION_PROMPTS[input_mode]},
        *sequence[feed.CONTENT_FIELD[input_mode]],
        {"type": "text", "text": DESCRIPTION_INSTRUCTION},
    ]


def validate_media(sequences: list[dict], runner: str, input_mode: str) -> None:
    missing: list[str] = []
    for sequence in sequences:
        for item in feed.content_for_runner(description_content(sequence, input_mode), runner):
            if item.get("type") == "video" and not Path(item["video"]).exists():
                missing.append(item["video"])
    if missing:
        sample = "\n".join(sorted(set(missing))[:20])
        raise FileNotFoundError(f"Missing video files:\n{sample}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ask a VLM to describe each material slide by slide."
    )
    parser.add_argument("--model-id", default=feed.DEFAULT_MODEL_ID)
    parser.add_argument(
        "--runner",
        choices=("auto", "qwen25", "qwen2", "qwen3", "gemma4", "qwen_omni", "minicpmo"),
        default="auto",
    )
    parser.add_argument("--sequence-index", type=Path, default=feed.SEQUENCE_INDEX)
    parser.add_argument(
        "--input-mode",
        choices=feed.INPUT_MODES,
        default="split",
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--print-sequences", action="store_true")
    parser.add_argument("--print-video-shapes", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--validate-media", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    runner = feed.resolve_runner(args.model_id, args.runner)
    feed.validate_supported_model(args.model_id, runner)
    output_path = args.output or default_output_path(args.model_id, args.input_mode)

    sequences = feed.load_jsonl(args.sequence_index)
    if args.limit is not None:
        sequences = sequences[: args.limit]
    if not sequences:
        raise FileNotFoundError(f"No sequences found in {args.sequence_index}")

    print(f"Found {len(sequences)} sequence(s).")
    print(f"Input mode: {args.input_mode}")
    print(f"Description prompt version: {PROMPT_VERSION}")
    if args.validate_media:
        validate_media(sequences, runner, args.input_mode)
    if args.validate_only:
        print("Validation passed.")
        return

    print(f"Loading model: {args.model_id} [{runner}]")
    model, processor = feed.load_model(args.model_id, runner)
    print("CUDA available:", feed.torch.cuda.is_available())
    print("Model device:", next(model.parameters()).device)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for sequence in sequences:
            source_video = sequence["source_video"]
            print(f"Running: {source_video}")
            content = description_content(sequence, args.input_mode)
            if args.print_sequences:
                preview = {
                    "source_video": source_video,
                    "runner": runner,
                    "input_mode": args.input_mode,
                    "content": feed.content_for_runner(content, runner),
                }
                print(json.dumps(preview, ensure_ascii=False, indent=2))

            try:
                response = feed.run_one_sequence(
                    model=model,
                    processor=processor,
                    content=content,
                    runner=runner,
                    max_new_tokens=args.max_new_tokens,
                    print_video_shapes=args.print_video_shapes,
                )
                record = {
                    "model": args.model_id,
                    "runner": runner,
                    "source_video": source_video,
                    "input_mode": args.input_mode,
                    "prompt_version": PROMPT_VERSION,
                    "max_new_tokens": args.max_new_tokens,
                    "content": feed.content_for_runner(content, runner),
                    "response": response,
                }
                preview = response[:500]
                print(f"Output preview ({len(preview)}/{len(response)} characters): {preview}")
            except Exception as exc:
                record = {
                    "model": args.model_id,
                    "runner": runner,
                    "source_video": source_video,
                    "input_mode": args.input_mode,
                    "prompt_version": PROMPT_VERSION,
                    "error": repr(exc),
                }
                print(f"Error on {source_video}: {exc}")
                traceback.print_exc()

            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            f.flush()
            print("-" * 80)

    print(f"Wrote descriptions to {output_path}")


if __name__ == "__main__":
    main()
