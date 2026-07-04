from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration


MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SEQUENCE_INDEX = PROJECT_ROOT / "processed" / "vlm_sequences" / "index.jsonl"
STATEMENT_FILE = PROJECT_ROOT / "code" / "task_statements.json"
RESULTS_DIR = PROJECT_ROOT / "results"

OUTPUT_FILE = RESULTS_DIR / "qwen2_5_vl_3b_sequence_ratings.jsonl"
STATEMENT_TYPES = ("target", "baseline")


def load_model() -> tuple[Qwen2_5_VLForConditionalGeneration, AutoProcessor]:
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_ID,
        torch_dtype="auto",
        device_map="auto",
    )
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    return model, processor


def load_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def normalize_statement_entry(source_video: str, entry: object) -> dict[str, str]:
    if isinstance(entry, str):
        return {"target": entry.strip(), "baseline": ""}

    if not isinstance(entry, dict):
        raise ValueError(
            f"Statement entry for {source_video} must be a string or an object "
            "with target/baseline fields."
        )

    unknown_keys = sorted(set(entry) - set(STATEMENT_TYPES))
    if unknown_keys:
        raise ValueError(
            f"Unknown statement fields for {source_video}: {', '.join(unknown_keys)}. "
            f"Allowed fields are: {', '.join(STATEMENT_TYPES)}."
        )

    return {
        statement_type: str(entry.get(statement_type, "")).strip()
        for statement_type in STATEMENT_TYPES
    }


def load_statements(path: Path) -> dict[str, dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(
            f"Cannot find statement file: {path}\n"
            "Create it from code/task_statements.json and fill target/baseline statements."
        )
    raw_statements = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw_statements, dict):
        raise ValueError(f"Statement file must contain a JSON object: {path}")
    return {
        str(source_video): normalize_statement_entry(str(source_video), entry)
        for source_video, entry in raw_statements.items()
    }


def rating_prompt(statement: str) -> str:
    return (
        "Based on the full sequence above, including the written text and the visual events, "
        "to what degree does the video suggest the following statement?\n\n"
        f"\"{statement}\"\n\n"
        "Answer with a single integer from 1 to 100.\n"
        "1 = not suggested at all.\n"
        "100 = very strongly suggested.\n"
        "Output only the number."
    )


def sequence_content(sequence: dict, statement: str) -> list[dict]:
    content = list(sequence["qwen_content"])
    content.append({"type": "text", "text": rating_prompt(statement)})
    return content


def print_sequence_job(sequence: dict, statement_type: str, statement: str) -> None:
    preview = {
        "source_video": sequence["source_video"],
        "statement_type": statement_type,
        "statement": statement,
        "content": sequence_content(sequence, statement),
    }
    print(json.dumps(preview, ensure_ascii=False, indent=2))


def validate_inputs(
    sequences: list[dict],
    statements: dict[str, dict[str, str]],
    statement_types: list[str],
    allow_missing_statements: bool,
) -> list[tuple[dict, str, str]]:
    jobs = []
    missing = []

    for sequence in sequences:
        source_video = sequence["source_video"]
        task_statements = statements.get(source_video, {})
        for statement_type in statement_types:
            statement = task_statements.get(statement_type, "")
            if statement:
                jobs.append((sequence, statement_type, statement))
            else:
                missing.append(f"{source_video} [{statement_type}]")

    if missing and not allow_missing_statements:
        missing_list = "\n".join(f"- {entry}" for entry in missing)
        raise ValueError(
            "Missing statements for these source videos:\n"
            f"{missing_list}\n\n"
            f"Fill them in {STATEMENT_FILE}, or rerun with --allow-missing-statements to skip them."
        )

    return jobs


def run_one_sequence(
    model: Qwen2_5_VLForConditionalGeneration,
    processor: AutoProcessor,
    content: list[dict],
    max_new_tokens: int,
    print_video_shapes: bool,
) -> str:
    messages = [{"role": "user", "content": content}]

    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    image_inputs, video_inputs = process_vision_info(messages)

    if print_video_shapes and video_inputs:
        print("video input shapes:", [tuple(video.shape) for video in video_inputs])

    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(model.device)

    with torch.inference_mode():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )

    generated_ids_trimmed = [
        output_ids[len(input_ids) :]
        for input_ids, output_ids in zip(inputs.input_ids, generated_ids)
    ]
    return processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0].strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Qwen on every prepared text/video sequence and collect 1-100 statement ratings."
    )
    parser.add_argument("--sequence-index", type=Path, default=SEQUENCE_INDEX)
    parser.add_argument("--statements", type=Path, default=STATEMENT_FILE)
    parser.add_argument("--output", type=Path, default=OUTPUT_FILE)
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--statement-types",
        nargs="+",
        choices=STATEMENT_TYPES,
        default=list(STATEMENT_TYPES),
        help="Which statement slots to run.",
    )
    parser.add_argument("--allow-missing-statements", action="store_true")
    parser.add_argument(
        "--print-sequences",
        action="store_true",
        help="Print the exact text/video sequence content for each runnable statement job.",
    )
    parser.add_argument("--print-video-shapes", action="store_true")
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Check sequence/statement coverage without loading the model.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    sequences = load_jsonl(args.sequence_index)
    if args.limit is not None:
        sequences = sequences[: args.limit]
    if not sequences:
        raise FileNotFoundError(f"No sequences found in {args.sequence_index}")

    statements = load_statements(args.statements)
    jobs = validate_inputs(
        sequences,
        statements,
        statement_types=args.statement_types,
        allow_missing_statements=args.allow_missing_statements,
    )
    if not jobs:
        raise ValueError("No sequences have statements to run.")

    print(f"Found {len(jobs)} runnable statement jobs.")
    if args.validate_only:
        if args.print_sequences:
            for sequence, statement_type, statement in jobs:
                print_sequence_job(sequence, statement_type, statement)
        print("Validation passed.")
        return

    print(f"Loading model: {MODEL_ID}")
    model, processor = load_model()
    print("CUDA available:", torch.cuda.is_available())
    print("Model device:", next(model.parameters()).device)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        for sequence, statement_type, statement in jobs:
            source_video = sequence["source_video"]
            print(f"Running: {source_video} [{statement_type}]")
            if args.print_sequences:
                print_sequence_job(sequence, statement_type, statement)

            try:
                content = sequence_content(sequence, statement)
                response = run_one_sequence(
                    model=model,
                    processor=processor,
                    content=content,
                    max_new_tokens=args.max_new_tokens,
                    print_video_shapes=args.print_video_shapes,
                )
                record = {
                    "model": MODEL_ID,
                    "source_video": source_video,
                    "statement_type": statement_type,
                    "statement": statement,
                    "response": response,
                    "content": content,
                }
                print(f"Output: {response}")
            except Exception as exc:
                record = {
                    "model": MODEL_ID,
                    "source_video": source_video,
                    "statement_type": statement_type,
                    "statement": statement,
                    "error": repr(exc),
                }
                print(f"Error on {source_video} [{statement_type}]: {exc}")

            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            f.flush()
            print("-" * 80)

    print(f"Wrote results to {args.output}")


if __name__ == "__main__":
    main()
