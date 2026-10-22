from __future__ import annotations

import argparse
import json
import traceback
from pathlib import Path

import feed_experiment as feed


PROMPT_VERSION = "3"


DESCRIPTION_PROMPT = """Describe the provided material as perceptual evidence only.
Do not answer any pragmatic inference question.
The material contains labeled text segments and/or labeled video clips. The labels are
navigation metadata, not part of the material itself.

Return valid JSON only, with this schema:
{
  "timeline": [
    {
      "order": 1,
      "type": "text",
      "text": "transcribed on-screen text or provided text"
    },
    {
      "order": 2,
      "type": "visual",
      "clip": 1,
      "description": "brief event description",
      "objects": ["shape/object names"],
      "colors": ["visible colors"],
      "locations": ["left/right/top/bottom/center if visible"],
      "event_type": "appear | disappear | fade in | fade out | move | change | unknown",
      "motion": "none | position changes | unknown"
    }
  ],
  "uncertain": ["anything you are not sure about"]
}

Rules:
- Use one flat timeline containing text and visual entries in their exact global order.
- Number order consecutively across the entire material; do not restart it for each clip.
- Copy material text into text entries without the navigation labels.
- Visual entries must contain only non-text events seen in video clips, or events explicitly
  introduced by an "Animation:" description when the material is text-only.
- Never copy or paraphrase narrative text into visual entries.
- Inspect every labeled video clip and include its clip number in each visual entry. Use null
  for clip when a visual event comes from an explicit text-only "Animation:" description.
- Place each visual entry immediately after the text or visual entry that preceded it.
- Do not infer motion unless object position changes across frames.
- If two adjacent events are hard to separate, describe them together but preserve order.
- Use [] when a field has no visible evidence."""


def default_output_path(model_id: str, input_mode: str) -> Path:
    mode = input_mode.replace("-", "_")
    return (
        feed.RESULTS_DIR
        / "descriptions"
        / f"{feed.model_slug(model_id)}_{mode}_descriptions.jsonl"
    )


def description_content(sequence: dict, input_mode: str) -> list[dict]:
    material = feed.base_content(sequence, input_mode)
    content = [{"type": "text", "text": DESCRIPTION_PROMPT}]
    text_number = 0
    video_number = 0

    for item in material:
        if item["type"] == "text":
            text_number += 1
            content.append(
                {
                    "type": "text",
                    "text": f"[Material text segment {text_number}]\n{item['text']}",
                }
            )
        elif item["type"] == "video":
            video_number += 1
            content.append(
                {
                    "type": "text",
                    "text": (
                        f"[Video clip {video_number} begins. Report its non-text visual "
                        "events in visual_events.]"
                    ),
                }
            )
            content.append(dict(item))
            content.append(
                {"type": "text", "text": f"[Video clip {video_number} ends.]"}
            )
        else:
            raise ValueError(f"Unsupported material item: {item}")

    content.append({"type": "text", "text": "[End of material. Return the JSON now.]"})
    return content


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
        description="Ask a VLM to produce structured descriptions of each material."
    )
    parser.add_argument("--model-id", default=feed.DEFAULT_MODEL_ID)
    parser.add_argument(
        "--runner",
        choices=("auto", "qwen25", "qwen2", "qwen3", "gemma4"),
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
