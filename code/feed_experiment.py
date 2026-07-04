from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import torch
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration


DEFAULT_MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"
DEFAULT_GEMMA_MODEL_ID = "google/gemma-4-12B-it"

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SEQUENCE_INDEX = PROJECT_ROOT / "processed" / "vlm_sequences" / "index.jsonl"
STATEMENT_FILE = PROJECT_ROOT / "code" / "task_statements.json"
RESULTS_DIR = PROJECT_ROOT / "results"

STATEMENT_TYPES = ("target", "baseline")
STATEMENT_TYPE_ORDER = {statement_type: i for i, statement_type in enumerate(STATEMENT_TYPES)}
PREMISE_ORDER = {
    "TargetPremise": 0,
    "ControlPremise": 1,
}
CONDITION_ORDER = {
    "Positive": 0,
    "Question": 0,
    "Negative": 1,
    "None": 1,
}


def model_slug(model_id: str) -> str:
    slug = model_id.lower().replace("/", "__")
    slug = re.sub(r"[^a-z0-9_.-]+", "_", slug)
    return slug.strip("_")


def default_output_path(model_id: str) -> Path:
    return RESULTS_DIR / f"{model_slug(model_id)}_sequence_ratings.jsonl"


def resolve_runner(model_id: str, runner: str) -> str:
    if runner != "auto":
        return runner

    normalized_model_id = model_id.lower()
    if "qwen2.5-vl" in normalized_model_id:
        return "qwen25"
    if "gemma-4" in normalized_model_id:
        return "gemma4"

    raise ValueError(
        "Could not infer a runner from --model-id. "
        "Pass --runner qwen25 or --runner gemma4 explicitly."
    )


def validate_supported_model(model_id: str, runner: str) -> None:
    normalized_model_id = model_id.lower()
    if runner == "qwen25" and "qwen2.5-vl" not in normalized_model_id:
        raise ValueError(
            "--runner qwen25 supports Qwen2.5-VL models only. "
            f"Received: {model_id}"
        )
    if runner == "gemma4" and "gemma-4" not in normalized_model_id:
        raise ValueError(
            "--runner gemma4 supports Gemma 4 models only. "
            f"Received: {model_id}"
        )


def load_qwen25_model(
    model_id: str,
) -> tuple[Qwen2_5_VLForConditionalGeneration, AutoProcessor]:
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype="auto",
        device_map="auto",
    )
    processor = AutoProcessor.from_pretrained(model_id)
    return model, processor


def load_gemma4_model(model_id: str):
    try:
        from transformers import AutoModelForMultimodalLM
    except ImportError as exc:
        raise ImportError(
            "Gemma 4 requires a recent transformers version with "
            "AutoModelForMultimodalLM. Try: pip install -U transformers"
        ) from exc

    model = AutoModelForMultimodalLM.from_pretrained(
        model_id,
        dtype="auto",
        device_map="auto",
    )
    processor = AutoProcessor.from_pretrained(model_id)
    return model, processor


def load_model(model_id: str, runner: str):
    if runner == "qwen25":
        return load_qwen25_model(model_id)
    if runner == "gemma4":
        return load_gemma4_model(model_id)
    raise ValueError(f"Unknown runner: {runner}")


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


def content_for_runner(content: list[dict], runner: str) -> list[dict]:
    if runner == "qwen25":
        return content
    if runner == "gemma4":
        gemma_content = []
        for item in content:
            if item["type"] == "text":
                gemma_content.append({"type": "text", "text": item["text"]})
            elif item["type"] == "video":
                gemma_content.append({"type": "video", "video": item["video"]})
            else:
                raise ValueError(f"Unsupported content item for Gemma 4: {item}")
        return gemma_content
    raise ValueError(f"Unknown runner: {runner}")


def print_sequence_job(
    sequence: dict,
    statement_type: str,
    statement: str,
    runner: str,
) -> None:
    content = sequence_content(sequence, statement)
    preview = {
        "source_video": sequence["source_video"],
        "statement_type": statement_type,
        "statement": statement,
        "runner": runner,
        "content": content_for_runner(content, runner),
    }
    print(json.dumps(preview, ensure_ascii=False, indent=2))


def parse_task_name(source_video: str) -> tuple[str, str | None, str | None]:
    task_name = Path(source_video).stem
    condition = None
    condition_match = re.search(r"\(([^)]+)\)$", task_name)
    if condition_match:
        condition = condition_match.group(1)
        task_name_without_condition = task_name[: condition_match.start()]
    else:
        task_name_without_condition = task_name

    premise = None
    family = task_name_without_condition
    for premise_name in PREMISE_ORDER:
        marker = f"_{premise_name}"
        if marker in task_name_without_condition:
            family = task_name_without_condition.replace(marker, "")
            premise = premise_name
            break

    return family, premise, condition


def task_sort_key(source_video: str) -> tuple[str, int, str, int, str]:
    task_name = Path(source_video).stem
    family, premise, condition = parse_task_name(source_video)

    condition_rank = CONDITION_ORDER.get(condition or "", 2)
    premise_rank = PREMISE_ORDER.get(premise or "", 2)

    return family, condition_rank, condition or "", premise_rank, task_name


def premise_label(source_video: str) -> str:
    _, premise, _ = parse_task_name(source_video)
    return premise or "Premise"


def sort_jobs(jobs: list[tuple[dict, str, str]]) -> list[tuple[dict, str, str]]:
    return sorted(
        jobs,
        key=lambda job: (
            *task_sort_key(job[0]["source_video"]),
            STATEMENT_TYPE_ORDER[job[1]],
        ),
    )


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

    return sort_jobs(jobs)


def run_one_qwen25_sequence(
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


def run_one_gemma4_sequence(
    model,
    processor: AutoProcessor,
    content: list[dict],
    max_new_tokens: int,
) -> str:
    messages = [{"role": "user", "content": content_for_runner(content, "gemma4")}]

    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        add_generation_prompt=True,
        enable_thinking=False,
    ).to(model.device)
    input_len = inputs["input_ids"].shape[-1]

    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )

    response = processor.decode(
        outputs[0][input_len:],
        skip_special_tokens=True,
    )
    return response.strip()


def run_one_sequence(
    model,
    processor: AutoProcessor,
    content: list[dict],
    runner: str,
    max_new_tokens: int,
    print_video_shapes: bool,
) -> str:
    if runner == "qwen25":
        return run_one_qwen25_sequence(
            model=model,
            processor=processor,
            content=content,
            max_new_tokens=max_new_tokens,
            print_video_shapes=print_video_shapes,
        )
    if runner == "gemma4":
        if print_video_shapes:
            print("video input shapes: not available for Gemma 4 runner")
        return run_one_gemma4_sequence(
            model=model,
            processor=processor,
            content=content,
            max_new_tokens=max_new_tokens,
        )
    raise ValueError(f"Unknown runner: {runner}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Qwen on every prepared text/video sequence and collect 1-100 statement ratings."
    )
    parser.add_argument(
        "--model-id",
        default=DEFAULT_MODEL_ID,
        help=(
            "Hugging Face model id. Supported runners: Qwen2.5-VL and "
            f"Gemma 4. Gemma default: {DEFAULT_GEMMA_MODEL_ID}."
        ),
    )
    parser.add_argument(
        "--runner",
        choices=("auto", "qwen25", "gemma4"),
        default="auto",
        help="Model adapter to use. auto infers from --model-id.",
    )
    parser.add_argument("--sequence-index", type=Path, default=SEQUENCE_INDEX)
    parser.add_argument("--statements", type=Path, default=STATEMENT_FILE)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="JSONL output path. Defaults to results/<model-id>_sequence_ratings.jsonl.",
    )
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
    runner = resolve_runner(args.model_id, args.runner)
    validate_supported_model(args.model_id, runner)
    output_path = args.output or default_output_path(args.model_id)

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
                print_sequence_job(sequence, statement_type, statement, runner)
        print("Validation passed.")
        return

    print(f"Loading model: {args.model_id} [{runner}]")
    model, processor = load_model(args.model_id, runner)
    print("CUDA available:", torch.cuda.is_available())
    print("Model device:", next(model.parameters()).device)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for sequence, statement_type, statement in jobs:
            source_video = sequence["source_video"]
            print(
                f"Running: {source_video} "
                f"[{premise_label(source_video)} / {statement_type}]"
            )
            if args.print_sequences:
                print_sequence_job(sequence, statement_type, statement, runner)

            try:
                content = sequence_content(sequence, statement)
                response = run_one_sequence(
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
                    "statement_type": statement_type,
                    "statement": statement,
                    "response": response,
                    "content": content_for_runner(content, runner),
                }
                print(f"Output: {response}")
            except Exception as exc:
                record = {
                    "model": args.model_id,
                    "runner": runner,
                    "source_video": source_video,
                    "statement_type": statement_type,
                    "statement": statement,
                    "error": repr(exc),
                }
                print(f"Error on {source_video} [{statement_type}]: {exc}")

            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            f.flush()
            print("-" * 80)

    print(f"Wrote results to {output_path}")


if __name__ == "__main__":
    main()
