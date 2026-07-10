from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import torch
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration, Qwen2VLForConditionalGeneration


DEFAULT_MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"
DEFAULT_GEMMA_MODEL_ID = "google/gemma-4-12B-it"

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SEQUENCE_INDEX = PROJECT_ROOT / "processed" / "vlm_sequences" / "index.jsonl"
STATEMENT_FILE = PROJECT_ROOT / "code" / "task_statements.json"
RESULTS_DIR = PROJECT_ROOT / "results"

STATEMENT_TYPES = ("target", "baseline")
INPUT_MODES = ("split", "pure-video", "pure-text")
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


def default_output_path(model_id: str, input_mode: str) -> Path:
    if input_mode == "split":
        return RESULTS_DIR / f"{model_slug(model_id)}_sequence_ratings.jsonl"
    mode_slug = input_mode.replace("-", "_")
    return RESULTS_DIR / f"{model_slug(model_id)}_{mode_slug}_sequence_ratings.jsonl"


def resolve_media_path(path_value: str) -> str:
    if re.match(r"^[a-z]+://", path_value):
        return path_value

    normalized = path_value.replace("\\", "/")
    portable_roots = (
        "processed/",
        "materials/",
    )
    for portable_root in portable_roots:
        if portable_root in normalized:
            suffix = normalized[normalized.index(portable_root) :]
            return str((PROJECT_ROOT / Path(suffix)).resolve())

    path = Path(path_value)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return str(path.resolve())


def resolve_runner(model_id: str, runner: str) -> str:
    if runner != "auto":
        return runner

    normalized_model_id = model_id.lower()
    if "qwen2.5-vl" in normalized_model_id:
        return "qwen25"
    if "qwen2-vl" in normalized_model_id:
        return "qwen2"
    if "gemma-4" in normalized_model_id:
        return "gemma4"

    raise ValueError(
        "Could not infer a runner from --model-id. "
        "Pass --runner qwen25, --runner qwen2, or --runner gemma4 explicitly."
    )


def validate_supported_model(model_id: str, runner: str) -> None:
    normalized_model_id = model_id.lower()
    if runner == "qwen25" and "qwen2.5-vl" not in normalized_model_id:
        raise ValueError(
            "--runner qwen25 supports Qwen2.5-VL models only. "
            f"Received: {model_id}"
        )
    if runner == "qwen2" and "qwen2-vl" not in normalized_model_id:
        raise ValueError(
            "--runner qwen2 supports Qwen2-VL models only. "
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


def load_qwen2_model(
    model_id: str,
) -> tuple[Qwen2VLForConditionalGeneration, AutoProcessor]:
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype="auto",
        device_map="auto",
    )
    processor = AutoProcessor.from_pretrained(model_id)
    return model, processor


def install_torchvision_read_video_fallback() -> None:
    """Provide torchvision.io.read_video for Gemma processors that still call it."""
    try:
        import torchvision
    except ImportError:
        return

    if hasattr(torchvision.io, "read_video"):
        return

    def read_video(
        filename: str,
        start_pts: float = 0,
        end_pts: float | None = None,
        pts_unit: str = "pts",
        output_format: str = "THWC",
    ):
        import av

        with av.open(filename) as container:
            stream = container.streams.video[0]
            fps = float(stream.average_rate) if stream.average_rate else 0.0
            frames = []
            for frame in container.decode(stream):
                timestamp = float(frame.pts or 0)
                if pts_unit == "sec":
                    timestamp *= float(stream.time_base)
                if timestamp < start_pts:
                    continue
                if end_pts is not None and timestamp > end_pts:
                    break
                frames.append(
                    torch.as_tensor(frame.to_ndarray(format="rgb24"), dtype=torch.uint8)
                )

        if not frames:
            raise RuntimeError(f"No video frames decoded from {filename}")

        video = torch.stack(frames)
        if output_format == "TCHW":
            video = video.permute(0, 3, 1, 2)
        elif output_format != "THWC":
            raise ValueError(f"Unsupported output_format for read_video fallback: {output_format}")

        audio = torch.empty((1, 0))
        return video, audio, {"video_fps": fps}

    torchvision.io.read_video = read_video


def load_gemma4_model(model_id: str):
    try:
        from transformers import AutoModelForMultimodalLM
    except ImportError as exc:
        raise ImportError(
            "Gemma 4 requires a recent transformers version with "
            "AutoModelForMultimodalLM. Try: pip install -U transformers"
        ) from exc

    install_torchvision_read_video_fallback()

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
    if runner == "qwen2":
        return load_qwen2_model(model_id)
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
        "Answer with a single integer from 0 to 100.\n"
        "0 = not suggested at all.\n"
        "100 = very strongly suggested.\n"
        "Output only the number."
    )


def base_content(
    sequence: dict,
    input_mode: str,
) -> list[dict]:
    field_by_mode = {
        "split": "qwen_content",
        "pure-video": "qwen_content_pure_video",
        "pure-text": "qwen_content_pure_text",
    }
    field = field_by_mode[input_mode]
    content = sequence.get(field)
    if not content:
        raise KeyError(
            f"Sequence for {sequence.get('source_video', '<unknown>')} is missing {field}. "
            "Rebuild sequences with: python code/build_vlm_sequences.py"
        )
    return list(content)


def sequence_content(
    sequence: dict,
    statement: str,
    input_mode: str,
) -> list[dict]:
    content = base_content(sequence, input_mode)
    content.append({"type": "text", "text": rating_prompt(statement)})
    return content


def content_for_runner(content: list[dict], runner: str) -> list[dict]:
    if runner in {"qwen25", "qwen2"}:
        runner_content = []
        for item in content:
            if item["type"] == "video":
                converted = dict(item)
                converted["video"] = resolve_media_path(str(item["video"]))
                runner_content.append(converted)
            else:
                runner_content.append(dict(item))
        return runner_content

    if runner == "gemma4":
        gemma_content = []
        for item in content:
            if item["type"] == "text":
                gemma_content.append({"type": "text", "text": item["text"]})
            elif item["type"] == "video":
                gemma_content.append(
                    {"type": "video", "video": resolve_media_path(str(item["video"]))}
                )
            else:
                raise ValueError(f"Unsupported content item for Gemma 4: {item}")
        return gemma_content
    raise ValueError(f"Unknown runner: {runner}")


def print_sequence_job(
    sequence: dict,
    statement_type: str,
    statement: str,
    runner: str,
    input_mode: str,
) -> None:
    content = sequence_content(sequence, statement, input_mode)
    preview = {
        "source_video": sequence["source_video"],
        "statement_type": statement_type,
        "statement": statement,
        "runner": runner,
        "input_mode": input_mode,
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


def validate_media_paths(
    jobs: list[tuple[dict, str, str]],
    runner: str,
    input_mode: str,
) -> None:
    checked = set()
    missing = []

    for sequence, _, statement in jobs:
        for item in content_for_runner(
            sequence_content(sequence, statement, input_mode),
            runner,
        ):
            if item["type"] != "video":
                continue
            video_path = item["video"]
            if video_path in checked or re.match(r"^[a-z]+://", video_path):
                continue
            checked.add(video_path)
            if not Path(video_path).exists():
                missing.append(video_path)

    if missing:
        missing_list = "\n".join(f"- {path}" for path in missing)
        raise FileNotFoundError(
            f"Missing {len(missing)} local video clip(s):\n{missing_list}"
        )

    print(f"Validated {len(checked)} local video clip path(s).")


def run_one_qwen_sequence(
    model: Qwen2_5_VLForConditionalGeneration | Qwen2VLForConditionalGeneration,
    processor: AutoProcessor,
    content: list[dict],
    runner: str,
    max_new_tokens: int,
    print_video_shapes: bool,
) -> str:
    messages = [{"role": "user", "content": content_for_runner(content, runner)}]

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
    if runner in {"qwen25", "qwen2"}:
        return run_one_qwen_sequence(
            model=model,
            processor=processor,
            content=content,
            runner=runner,
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
            f"Qwen2-VL, and Gemma 4. Gemma default: {DEFAULT_GEMMA_MODEL_ID}."
        ),
    )
    parser.add_argument(
        "--runner",
        choices=("auto", "qwen25", "qwen2", "gemma4"),
        default="auto",
        help="Model adapter to use. auto infers from --model-id.",
    )
    parser.add_argument("--sequence-index", type=Path, default=SEQUENCE_INDEX)
    parser.add_argument("--statements", type=Path, default=STATEMENT_FILE)
    parser.add_argument(
        "--input-mode",
        choices=INPUT_MODES,
        default="split",
        help=(
            "split uses text tokens plus split animation clips; pure-video feeds each original "
            "materials/animations video; pure-text replaces every slide, including animation "
            "clips, with text from the RTF story file."
        ),
    )
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
    parser.add_argument(
        "--validate-media",
        action="store_true",
        help="Check that all local video clip paths exist after runtime path conversion.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    runner = resolve_runner(args.model_id, args.runner)
    validate_supported_model(args.model_id, runner)
    output_path = args.output or default_output_path(args.model_id, args.input_mode)

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
    print(f"Input mode: {args.input_mode}")
    if args.validate_media:
        validate_media_paths(jobs, runner, args.input_mode)

    if args.validate_only:
        if args.print_sequences:
            for sequence, statement_type, statement in jobs:
                print_sequence_job(
                    sequence,
                    statement_type,
                    statement,
                    runner,
                    args.input_mode,
                )
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
                print_sequence_job(
                    sequence,
                    statement_type,
                    statement,
                    runner,
                    args.input_mode,
                )

            try:
                content = sequence_content(sequence, statement, args.input_mode)
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
                    "input_mode": args.input_mode,
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
                    "input_mode": args.input_mode,
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
