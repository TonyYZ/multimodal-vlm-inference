from __future__ import annotations

import argparse
import inspect
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
RESULTS_DIR = PROJECT_ROOT / "results" / "animation" / "inference" / "ratings"

STATEMENT_TYPES = ("target", "baseline")
# Greedy by default; --samples switches to pure temperature sampling (no top-k/top-p truncation).
GENERATION = {"do_sample": False}
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
    if "qwen2.5-omni" in normalized_model_id or "qwen3-omni" in normalized_model_id:
        return "qwen_omni"
    if "minicpm-o-4_5" in normalized_model_id:
        return "minicpmo"
    if "qwen3-vl" in normalized_model_id or "qwen3.8" in normalized_model_id:
        return "qwen3"
    if "qwen2.5-vl" in normalized_model_id:
        return "qwen25"
    if "qwen2-vl" in normalized_model_id:
        return "qwen2"
    if "gemma-4" in normalized_model_id:
        return "gemma4"

    raise ValueError(
        "Could not infer a runner from --model-id. "
        "Pass --runner qwen25, qwen2, qwen3, gemma4, qwen_omni, or minicpmo explicitly."
    )


def validate_supported_model(model_id: str, runner: str) -> None:
    normalized_model_id = model_id.lower()
    if runner == "qwen_omni" and not any(name in normalized_model_id for name in ("qwen2.5-omni", "qwen3-omni")):
        raise ValueError("--runner qwen_omni requires a Qwen2.5-Omni or Qwen3-Omni checkpoint.")
    if runner == "minicpmo" and "minicpm-o-4_5" not in normalized_model_id:
        raise ValueError("--runner minicpmo supports MiniCPM-o 4.5 only.")
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
    if runner == "qwen3" and not (
        "qwen3-vl" in normalized_model_id or "qwen3.8" in normalized_model_id
    ):
        raise ValueError(
            "--runner qwen3 supports Qwen3-VL and Qwen3.8 models only. "
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


def load_qwen3_model(model_id: str):
    try:
        from transformers import AutoModelForMultimodalLM
    except ImportError as exc:
        raise ImportError(
            "Qwen3-VL/Qwen3.8 requires a recent transformers version with "
            "AutoModelForMultimodalLM. Reinstall from requirements.txt or upgrade transformers."
        ) from exc

    load_kwargs = {"dtype": "auto", "device_map": "auto"}
    try:
        model = AutoModelForMultimodalLM.from_pretrained(model_id, **load_kwargs)
    except TypeError:
        load_kwargs["torch_dtype"] = load_kwargs.pop("dtype")
        model = AutoModelForMultimodalLM.from_pretrained(model_id, **load_kwargs)
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
    if runner == "qwen_omni":
        from omni_runners import load_qwen_omni

        return load_qwen_omni(model_id)
    if runner == "minicpmo":
        from omni_runners import load_minicpmo

        return load_minicpmo(model_id)
    if runner == "qwen25":
        return load_qwen25_model(model_id)
    if runner == "qwen2":
        return load_qwen2_model(model_id)
    if runner == "qwen3":
        return load_qwen3_model(model_id)
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


# Field of the sequence index (built by build_vlm_sequences.py) holding the material for each input mode.
CONTENT_FIELD = {
    "split": "qwen_content",
    "pure-video": "qwen_content_pure_video",
    "pure-text": "qwen_content_pure_text",
}


def sequence_content(sequence: dict, statement: str, input_mode: str) -> list[dict]:
    """Rating prompt: "Watch the following video." + material + question.
    The emoji pilot sets its own question (rating_prompt_intro) and gets no lead-in."""
    material = sequence[CONTENT_FIELD[input_mode]]
    intro = sequence.get("rating_prompt_intro")
    has_video = any(item["type"] == "video" for item in material)
    lead = [{"type": "text", "text": "Watch the following video."}] if has_video and not intro else []
    question = (
        f"{intro or 'To what degree do you infer the following?'}\n\n"
        f"\"{statement}\"\n\n"
        "Answer with a single integer from 0 to 100.\n"
        "0 = not suggested at all.\n"
        "100 = very strongly suggested.\n"
        "Output only the number."
    )
    return [*lead, *material, {"type": "text", "text": question}]


def content_for_runner(content: list[dict], runner: str) -> list[dict]:
    if runner in {"qwen25", "qwen2", "qwen3", "qwen_omni", "minicpmo"}:
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


def apply_chat_template(
    processor: AutoProcessor,
    messages: list[dict],
    runner: str,
) -> str:
    kwargs = {
        "tokenize": False,
        "add_generation_prompt": True,
    }
    if runner == "qwen3":
        kwargs["enable_thinking"] = False

    try:
        return processor.apply_chat_template(messages, **kwargs)
    except TypeError:
        kwargs.pop("enable_thinking", None)
        return processor.apply_chat_template(messages, **kwargs)


def supported_vision_kwargs(runner: str) -> dict:
    parameters = inspect.signature(process_vision_info).parameters
    kwargs = {}
    if runner == "qwen3" and "image_patch_size" in parameters:
        kwargs["image_patch_size"] = 16
    if "return_video_kwargs" in parameters:
        kwargs["return_video_kwargs"] = True
    if "return_video_metadata" in parameters:
        kwargs["return_video_metadata"] = True
    return kwargs


def warn_old_qwen3_video_utils() -> None:
    if getattr(warn_old_qwen3_video_utils, "shown", False):
        return
    warn_old_qwen3_video_utils.shown = True
    print(
        "Warning: this qwen-vl-utils version does not provide Qwen3 video metadata. "
        "The run can continue, but upgrading qwen-vl-utils is recommended for Qwen3 video jobs."
    )


def process_qwen_vision_info(messages: list[dict], runner: str):
    requested_kwargs = supported_vision_kwargs(runner)
    result = process_vision_info(messages, **requested_kwargs)
    if len(result) == 3:
        image_inputs, video_inputs, video_kwargs = result
    else:
        image_inputs, video_inputs = result
        video_kwargs = {}

    video_metadata = None
    if video_inputs:
        first_video = video_inputs[0]
        if isinstance(first_video, tuple) and len(first_video) == 2:
            videos, metadata = zip(*video_inputs)
            video_inputs = list(videos)
            video_metadata = list(metadata)
        elif runner == "qwen3":
            warn_old_qwen3_video_utils()

    sampled_fps = video_kwargs.get("fps")
    fps_values = None
    if sampled_fps is not None:
        try:
            fps_values = [float(value) for value in sampled_fps]
        except TypeError:
            fps_values = [float(sampled_fps)] * len(video_inputs or [])

    requested_fps = [
        float(item["fps"])
        for message in messages
        for item in message.get("content", [])
        if item.get("type") == "video" and item.get("fps") is not None
    ]

    if video_inputs:
        normalized_metadata = []
        for index, video in enumerate(video_inputs):
            metadata = video_metadata[index] if video_metadata else None
            duration = (
                metadata.get("duration")
                if isinstance(metadata, dict)
                else getattr(metadata, "duration", None)
            )
            if not duration and metadata is not None:
                source_frames = (
                    metadata.get("total_num_frames")
                    if isinstance(metadata, dict)
                    else getattr(metadata, "total_num_frames", None)
                )
                source_fps = (
                    metadata.get("fps")
                    if isinstance(metadata, dict)
                    else getattr(metadata, "fps", None)
                )
                if source_frames and source_fps:
                    duration = float(source_frames) / float(source_fps)
            if fps_values and len(fps_values) == len(video_inputs):
                fps = fps_values[index]
            elif duration:
                fps = len(video) / float(duration)
            elif len(requested_fps) == len(video_inputs):
                fps = requested_fps[index]
            else:
                continue

            frame_count = len(video)
            normalized_metadata.append(
                {
                    "total_num_frames": frame_count,
                    "fps": fps,
                    "width": int(video.shape[-1]),
                    "height": int(video.shape[-2]),
                    "duration": frame_count / fps,
                    "video_backend": "qwen-vl-utils",
                    "frames_indices": list(range(frame_count)),
                }
            )

        if len(normalized_metadata) == len(video_inputs):
            video_metadata = normalized_metadata

    return image_inputs, video_inputs, video_metadata, video_kwargs


def run_one_qwen_sequence(
    model,
    processor: AutoProcessor,
    content: list[dict],
    runner: str,
    max_new_tokens: int,
    print_video_shapes: bool,
) -> str:
    messages = [{"role": "user", "content": content_for_runner(content, runner)}]

    text = apply_chat_template(processor, messages, runner)
    image_inputs, video_inputs, video_metadata, video_kwargs = process_qwen_vision_info(
        messages,
        runner,
    )

    if print_video_shapes and video_inputs:
        print("video input shapes:", [tuple(video.shape) for video in video_inputs])
        if video_metadata is not None:
            print(
                "video sampled fps:",
                [
                    getattr(metadata, "sampled_fps", None)
                    if not isinstance(metadata, dict)
                    else metadata.get("fps")
                    for metadata in video_metadata
                ],
            )

    processor_inputs = {
        "text": [text],
        "images": image_inputs,
        "videos": video_inputs,
        "padding": True,
        "return_tensors": "pt",
    }
    if video_inputs is not None:
        video_kwargs = {**video_kwargs, "do_sample_frames": False}
        if runner == "qwen3":
            processor_inputs["do_resize"] = False
    if video_metadata is not None:
        processor_inputs["video_metadata"] = video_metadata
        video_kwargs.pop("fps", None)
    elif isinstance(video_kwargs.get("fps"), (list, tuple)):
        print(
            "Warning: per-video FPS metadata is unavailable; the processor will use its "
            "default timing metadata. Upgrade qwen-vl-utils to preserve per-video timing."
        )
        video_kwargs.pop("fps")

    if video_kwargs:
        inputs = processor(
            **processor_inputs,
            **video_kwargs,
        ).to(model.device)
    else:
        inputs = processor(**processor_inputs).to(model.device)

    if print_video_shapes and video_inputs:
        pixels = inputs.get("pixel_values_videos")
        grid = inputs.get("video_grid_thw")
        token_id = getattr(model.config, "video_token_id", None)
        print("processed video tensor shape:", tuple(pixels.shape) if pixels is not None else None)
        print("processed video grid:", grid.tolist() if grid is not None else None)
        print("video token count:", int((inputs.input_ids == token_id).sum()) if token_id is not None else "unknown")

    if GENERATION.get("score_distribution"):
        return score_distribution(model, processor.tokenizer, inputs)
    with torch.inference_mode():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            **GENERATION,
        )

    responses = processor.batch_decode(
        generated_ids[:, inputs.input_ids.shape[-1]:],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    return [r.strip() for r in responses] if GENERATION["do_sample"] else responses[0].strip()


def score_distribution(model, tokenizer, inputs) -> dict:
    """Exact probability of each rating 0-100 as the first thing the model writes.

    Ratings are scored digit by digit under teacher forcing:
    P(85) = P('8') * P('5' | '8') * P(no further digit | '85'). The prompt is encoded once;
    each digit prefix reuses a copy of its cache, so only 1-3 new tokens are processed per call.
    """
    import copy
    import math

    digit_ids = []
    for digit in "0123456789":
        ids = tokenizer.encode(digit, add_special_tokens=False)
        if len(ids) != 1:
            raise ValueError(f"Digit {digit!r} is not a single token: {ids}")
        digit_ids.append(ids[0])
    for number in ("85", "100"):
        if tokenizer.encode(number, add_special_tokens=False) != [digit_ids[int(c)] for c in number]:
            raise ValueError("Tokenizer does not split numbers into single-digit tokens.")
    digit_tensor = torch.tensor(digit_ids, device=model.device)
    settings = dict(max_new_tokens=1, do_sample=False, output_logits=True, return_dict_in_generate=True)

    with torch.inference_mode():
        first = model.generate(**inputs, **settings)
    prompt_cache = first.past_key_values
    seq_len = inputs["input_ids"].shape[-1]
    log_probs = {"": torch.log_softmax(first.logits[0][0].float(), -1)}

    def next_log_probs(prefix: str):
        if prefix not in log_probs:
            forced = torch.tensor([[digit_ids[int(c)] for c in prefix]], device=model.device)
            extended = {}
            for key, value in inputs.items():
                # Keep only per-token inputs (ids, attention mask, token type ids), extended by the
                # forced digits. Image/video features are already in the cache and must not be resent.
                if torch.is_tensor(value) and value.dim() == 2 and value.shape[-1] == seq_len:
                    pad = forced if key == "input_ids" else (
                        torch.ones_like(forced) if "attention" in key else torch.zeros_like(forced))
                    extended[key] = torch.cat([value, pad.to(value.dtype)], dim=-1)
            with torch.inference_mode():
                out = model.generate(**extended, past_key_values=copy.deepcopy(prompt_cache), **settings)
            log_probs[prefix] = torch.log_softmax(out.logits[0][0].float(), -1)
        return log_probs[prefix]

    distribution = []
    for rating in range(101):
        text = str(rating)
        log_p = sum(next_log_probs(text[:i])[digit_ids[int(c)]].item() for i, c in enumerate(text))
        end = 1.0 - next_log_probs(text).exp()[digit_tensor].sum().item()
        distribution.append(math.exp(log_p) * max(end, 0.0))
    mass = sum(distribution)
    expected = sum(k * p for k, p in enumerate(distribution)) / mass if mass else float("nan")
    variance = sum((k - expected) ** 2 * p for k, p in enumerate(distribution)) / mass if mass else float("nan")
    return {"distribution": distribution, "mass": mass, "expected": expected,
            "sd": math.sqrt(variance), "mode": int(max(range(101), key=distribution.__getitem__))}


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

    if GENERATION.get("score_distribution"):
        return score_distribution(model, processor.tokenizer, inputs)
    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            **GENERATION,
        )

    responses = processor.batch_decode(outputs[:, input_len:], skip_special_tokens=True)
    return [r.strip() for r in responses] if GENERATION["do_sample"] else responses[0].strip()


def run_one_sequence(
    model,
    processor: AutoProcessor,
    content: list[dict],
    runner: str,
    max_new_tokens: int,
    print_video_shapes: bool,
) -> str:
    if (GENERATION["do_sample"] or GENERATION.get("score_distribution")) and runner in {"qwen_omni", "minicpmo"}:
        raise NotImplementedError("--samples/--score-distribution are only implemented for Qwen-VL and Gemma 4.")
    if runner == "qwen_omni":
        from omni_runners import run_qwen_omni_animation

        return run_qwen_omni_animation(
            model, processor, content_for_runner(content, runner), max_new_tokens, print_video_shapes,
        )
    if runner == "minicpmo":
        from omni_runners import run_minicpmo

        return run_minicpmo(model, content_for_runner(content, runner), max_new_tokens)
    if runner in {"qwen25", "qwen2", "qwen3"}:
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
            "Hugging Face model id. Supported runners: Qwen2.5-VL, Qwen2-VL, "
            f"Qwen3-VL/Qwen3.8, Gemma 4, Qwen Omni, and MiniCPM-o 4.5. Gemma default: {DEFAULT_GEMMA_MODEL_ID}."
        ),
    )
    parser.add_argument(
        "--runner",
        choices=("auto", "qwen25", "qwen2", "qwen3", "gemma4", "qwen_omni", "minicpmo"),
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
        help="JSONL output path. Defaults to results/animation/inference/ratings/<model-id>_sequence_ratings.jsonl.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--samples", type=int, default=0,
                        help="Sampled ratings per statement (0 = one greedy rating).")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--sample-batch", type=int, default=10,
                        help="Samples generated per forward pass.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--score-distribution", action="store_true",
                        help="Write the exact 0-100 rating distribution from the logits instead of generating.")
    parser.add_argument("--resume", action="store_true",
                        help="Keep complete error-free statements in --output and rerun the rest.")
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

    sequences = load_jsonl(args.sequence_index)
    if args.limit is not None:
        sequences = sequences[: args.limit]
    if not sequences:
        raise FileNotFoundError(f"No sequences found in {args.sequence_index}")

    is_emoji = all(
        "emoji_study" in sequence["source_video"].replace("\\", "/").split("/")
        for sequence in sequences
    )
    if is_emoji:
        default_output = (
            PROJECT_ROOT / "results" / "emoji" / "inference" / "ratings"
            / f"{model_slug(args.model_id)}_emoji_pilot.jsonl"
        )
    else:
        default_output = default_output_path(args.model_id, args.input_mode)
    output_path = args.output or default_output

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

    if args.score_distribution:
        if args.samples:
            raise ValueError("Use either --samples or --score-distribution.")
        GENERATION["score_distribution"] = True
        if args.output is None:
            output_path = output_path.with_name(f"{output_path.stem}_logit_scores.jsonl")
    if args.samples:
        # Override every default that reshapes the distribution (Qwen2.5-VL ships repetition_penalty=1.05).
        GENERATION.update(do_sample=True, temperature=args.temperature, top_p=1.0, top_k=0,
                          repetition_penalty=1.0, min_p=None, typical_p=1.0)
        torch.manual_seed(args.seed)
        if args.output is None:
            output_path = output_path.with_name(
                f"{output_path.stem}_T{args.temperature:g}_n{args.samples}.jsonl")
        print(f"Sampling {args.samples} ratings per statement at temperature {args.temperature}")

    print(f"Loading model: {args.model_id} [{runner}]")
    model, processor = load_model(args.model_id, runner)
    print("CUDA available:", torch.cuda.is_available())
    print("Model device:", next(model.parameters()).device)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if args.resume and output_path.exists():
        # Keep complete, error-free statements; everything else is rerun from scratch.
        old = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        counts = {}
        for row in old:
            if "error" not in row:
                key = (row["source_video"], row["statement_type"])
                counts[key] = counts.get(key, 0) + 1
        done = {key for key, n in counts.items() if n == max(args.samples, 1)}
        kept = [row for row in old if (row["source_video"], row["statement_type"]) in done]
        output_path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in kept), encoding="utf-8")
        print(f"Resuming: {len(done)} statements already complete.")
    sample_batch = args.sample_batch
    with output_path.open("a" if args.resume else "w", encoding="utf-8") as f:
        for sequence, statement_type, statement in jobs:
            source_video = sequence["source_video"]
            if (source_video, statement_type) in done:
                continue
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

            base = {
                "model": args.model_id,
                "runner": runner,
                "source_video": source_video,
                "input_mode": args.input_mode,
                "statement_type": statement_type,
                "statement": statement,
            }
            if args.samples:
                base.update(temperature=args.temperature, seed=args.seed)
            records = []
            try:
                content = sequence_content(sequence, statement, args.input_mode)
                if args.score_distribution:
                    result = run_one_sequence(model=model, processor=processor, content=content, runner=runner,
                                              max_new_tokens=args.max_new_tokens,
                                              print_video_shapes=args.print_video_shapes)
                    records = [dict(base, **result, content=content_for_runner(content, runner))]
                    print(f"Expected {result['expected']:.1f} (sd {result['sd']:.1f}), "
                          f"mode {result['mode']}, number mass {result['mass']:.3f}")
                    for record in records:
                        f.write(json.dumps(record, ensure_ascii=False) + "\n")
                    f.flush()
                    continue
                responses = []
                while len(responses) < max(args.samples, 1):
                    if args.samples:
                        GENERATION["num_return_sequences"] = min(sample_batch, args.samples - len(responses))
                    try:
                        response = run_one_sequence(
                            model=model,
                            processor=processor,
                            content=content,
                            runner=runner,
                            max_new_tokens=args.max_new_tokens,
                            print_video_shapes=args.print_video_shapes,
                        )
                    except torch.cuda.OutOfMemoryError:
                        # Long inputs are copied once per sample; retry the same chunk with fewer samples.
                        torch.cuda.empty_cache()
                        if not args.samples or sample_batch == 1:
                            raise
                        sample_batch = max(1, sample_batch // 2)
                        print(f"Out of memory; sample batch reduced to {sample_batch}")
                        continue
                    responses.extend(response if isinstance(response, list) else [response])
                for index, response in enumerate(responses):
                    record = dict(base, response=response)
                    if args.samples:
                        record["sample"] = index
                    if index == 0:
                        record["content"] = content_for_runner(content, runner)
                    records.append(record)
                print(f"Output: {responses[:10]}")
            except Exception as exc:
                records = [dict(base, error=repr(exc))]
                print(f"Error on {source_video} [{statement_type}]: {exc}")

            for record in records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
            f.flush()
            print("-" * 80)

    print(f"Wrote results to {output_path}")


if __name__ == "__main__":
    main()
