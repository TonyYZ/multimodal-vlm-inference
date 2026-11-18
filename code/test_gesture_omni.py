from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import shutil
import subprocess
import tempfile
import traceback
from pathlib import Path

from extract_gold_descriptions import extract_gold


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_ID = "Qwen/Qwen2.5-Omni-7B"
DEFAULT_VIDEO_DIR = PROJECT_ROOT / "materials" / "gestures" / "OSF Stimuli Anonymized"
DEFAULT_STORIES = PROJECT_ROOT / "materials" / "stories" / "TSC_EXP_1_Stories.rtf"
PIPELINE_VERSION = "gesture-av-2"
MODEL_RUNNERS = {
    "Qwen/Qwen2.5-Omni-3B": "qwen25_omni",
    "Qwen/Qwen2.5-Omni-7B": "qwen25_omni",
    "Qwen/Qwen3-Omni-30B-A3B-Instruct": "qwen3_omni",
    "google/gemma-4-12B-it": "gemma4",
    "openbmb/MiniCPM-o-4_5": "minicpmo",
}
ANONYMIZATION_NOTE = (
    "The video shows a person speaking and making hand gestures. "
    "The blur over the person's face is an anonymization mask. "
    "Do not infer hidden facial details."
)
DESCRIPTION_PROMPT_VERSION = "gesture-av-2-mask-note"
SANITY_PROMPT_VERSION = "gesture-sanity-2-mask-note"
RATING_PROMPT_VERSION = "gesture-rating-2-mask-note"
SANITY_PROMPT = """Inspect the video and its audio. Write two short sections, not JSON:
Speech: transcribe every spoken word literally, in order. Do not replace silent
gestures with words. Mark unclear words [inaudible].
Gestures: briefly describe the visible hand movements in chronological order,
including direction, repetitions and whether one or both hands are used.
Use the viewer's left/right. Describe movements, not their inferred meanings.
Do not add timestamps, invent details, or answer an inference question."""
DESCRIPTION_PROMPT = """Describe only what you can hear and see in the supplied video.
This is a perception check, not an inference-rating task.
Transcribe the spoken English literally. Do not paraphrase, fix grammar, or insert
words to replace silent gestures. If an utterance stops before a silent gesture,
transcribe only the words heard; describe the gesture separately.
Mark unclear speech as [inaudible]. Do not use outside knowledge to fill gaps.
Describe all visible hand gestures, their movement paths, pointing direction,
hand configuration, one/two-handedness and repetitions where distinguishable.
Use viewer-left/viewer-right (the viewer's perspective), not the actor's left/right.
Keep gesture descriptions physical; do not substitute inferred meanings for
observed hand movements.

Return valid JSON only:
{
  "timeline": [
    {"order": 1, "type": "speech", "start_s": null, "end_s": null,
     "text": "literal words heard"},
    {"order": 2, "type": "gesture", "start_s": null, "end_s": null,
     "description": "visible hand configuration and movement",
     "hands": "one | two | unclear", "location": "viewer-left/right/center",
     "motion": "observed movement path", "repetitions": null}
  ],
  "uncertain": ["specific unclear words or visual details"]
}

Use one flat timeline, ordered by onset, with consecutive order numbers.
Include speech throughout the video, not only its final sentence.
Include gestures during speech as well as gestures during silent gaps.
Speech and gesture entries may overlap in time; do not force them into separate
non-overlapping stages. Approximate seconds are optional: use null if unsure.
For actual on-screen words, use type "on_screen_text" and a "text" field;
do not label spoken words as on-screen text. Ignore filenames and task labels.
Merging or splitting events is acceptable if content and chronology are preserved.
Do not mention target/baseline inferences or give an endorsement rating."""


def parse_description(response: str) -> dict:
    from judge_descriptions import extract_json

    parsed = extract_json(response)
    if not parsed or not isinstance(parsed.get("timeline"), list) or not parsed["timeline"]:
        raise ValueError("Description must contain a nonempty JSON timeline.")
    if not isinstance(parsed.get("uncertain"), list):
        raise ValueError("Description must contain an uncertain list.")
    for order, event in enumerate(parsed["timeline"], 1):
        if not isinstance(event, dict) or type(event.get("order")) is not int or event["order"] != order:
            raise ValueError("Timeline orders must be consecutive integers starting at 1.")
        kind = event.get("type")
        if kind not in {"speech", "gesture", "on_screen_text"}:
            raise ValueError(f"Unknown timeline event type: {kind}")
        fields = ("description", "hands", "location", "motion") if kind == "gesture" else ("text",)
        if any(not isinstance(event.get(field), str) or not event[field].strip() for field in fields):
            raise ValueError(f"Missing content in timeline event {order}.")
        for field in ("start_s", "end_s"):
            value = event.get(field)
            if field not in event or (value is not None and (type(value) not in (int, float) or value < 0)):
                raise ValueError(f"{field} must be a nonnegative number or null.")
        if event["start_s"] is not None and event["end_s"] is not None and event["end_s"] < event["start_s"]:
            raise ValueError("An event cannot end before it starts.")
    return parsed


def audit_inputs(video: Path, inputs, audit: dict, *, audio=None, sample_rate=None) -> None:
    import numpy as np
    import torch

    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration:stream=codec_type,duration",
         "-of", "json", str(video)], capture_output=True, text=True, check=True,
    )
    source = json.loads(probe.stdout)
    audit["source_duration_s"] = float(source["format"]["duration"])
    audit["source_streams"] = source["streams"]
    tensors = {key: value for key, value in inputs.items() if isinstance(value, torch.Tensor)}
    audit["input_tensors"] = {
        key: {"shape": list(value.shape), "dtype": str(value.dtype)} for key, value in tensors.items()
    }
    visual = [key for key, value in tensors.items() if "pixel" in key and value.is_floating_point()]
    acoustic = [key for key, value in tensors.items()
                if ("input_features" in key or "audio_values" in key) and value.is_floating_point()]
    if not visual or not acoustic:
        raise RuntimeError(f"Missing processed modality: visual={visual}, audio={acoustic}")
    for key in visual + acoustic:
        value = tensors[key]
        if not value.numel() or not torch.isfinite(value).all() or not value.abs().max() > 0:
            raise RuntimeError(f"Empty, nonfinite or entirely zero input tensor: {key}")
    if audio is not None:
        waveform = np.asarray(audio)
        audit["audio_sample_rate"] = sample_rate
        audit["audio_duration_s"] = len(waveform) / sample_rate
        audit["audio_rms"] = float(np.sqrt(np.mean(waveform.astype(np.float64) ** 2)))
        if not np.isfinite(waveform).all() or audit["audio_rms"] <= 0:
            raise RuntimeError("Decoded audio is empty, nonfinite or silent.")
        audio_stream = next(s for s in source["streams"] if s["codec_type"] == "audio")
        expected_duration = float(audio_stream.get("duration", audit["source_duration_s"]))
        if abs(audit["audio_duration_s"] - expected_duration) > 0.5:
            raise RuntimeError("Decoded audio duration does not match the source audio stream.")
    audit["modality_checks_passed"] = True


def audit_generation(inputs, sequences, budget: int, audit: dict | None) -> None:
    if audit is None:
        return
    import torch

    length = inputs.input_ids.shape[1]
    audit["prompt_prefix_matches"] = bool(torch.equal(sequences[:, :length], inputs.input_ids))
    audit["input_tokens"] = length
    audit["generated_tokens"] = sequences.shape[1] - length
    audit["token_budget_reached"] = audit["generated_tokens"] >= budget
    if not audit["prompt_prefix_matches"]:
        raise RuntimeError("Generated sequence has no matching prompt prefix; answer trimming is unsafe.")

def model_slug(model_id: str) -> str:
    slug = model_id.lower().replace("/", "__")
    slug = re.sub(r"[^a-z0-9_.-]+", "_", slug)
    return slug.strip("_")


def rating_prompt(statement: str) -> str:
    return (
        "To what degree do you infer the following?\n\n"
        f"\"{statement}\"\n\n"
        "Answer with a single integer from 0 to 100.\n"
        "0 = not suggested at all.\n"
        "100 = very strongly suggested.\n"
        "Output only the number."
    )


def build_conversation(
    video_path: Path,
    statement: str,
    task: str = "rate",
) -> list[dict]:
    return [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": (
                        "You are a multimodal assistant. Answer using only the provided video, "
                        "including visible gestures and spoken content. " + ANONYMIZATION_NOTE
                    ),
                }
            ],
        },
        {
            "role": "user",
            "content": [
                # Lead-in before the video for the rating task only (describe/sanity keep their own prompts).
                *([{"type": "text", "text": "Watch the following video."}] if task == "rate" else []),
                {"type": "video", "video": str(video_path.resolve())},
                {"type": "text", "text": (
                    DESCRIPTION_PROMPT if task == "describe" else
                    SANITY_PROMPT if task == "sanity" else rating_prompt(statement)
                )},
            ],
        },
    ]


def prepare_gemma_video_processor(processor) -> None:
    video_processor = processor.video_processor
    video_processor.num_frames = None
    if getattr(video_processor, "_gesture_index_guard", False):
        return
    original_sampler = video_processor.sample_frames

    def sample_frames(metadata, **kwargs):
        indices = original_sampler(metadata=metadata, **kwargs)
        # Float-step arange can produce one extra index equal to the frame count.
        return indices[(indices >= 0) & (indices < metadata.total_num_frames)]

    video_processor.sample_frames = sample_frames
    video_processor._gesture_index_guard = True


def run_one(
    model,
    processor,
    video_path: Path,
    statement: str,
    max_new_tokens: int,
    use_audio_in_video: bool,
    runner: str = "qwen25_omni",
    fps: float = 3.0,
    max_pixels: int = 307200,
    task: str = "rate",
    audit: dict | None = None,
) -> tuple[str, list[dict]]:
    import torch

    conversation = build_conversation(video_path, statement, task)
    conversation[1]["content"][0].update(fps=fps, max_pixels=max_pixels)
    if runner == "minicpmo":
        from omni_runners import run_minicpmo

        response = run_minicpmo(
            model, conversation[1]["content"], max_new_tokens,
            system_message=conversation[0]["content"][0]["text"],
            use_audio=use_audio_in_video, audit=audit,
        )
        return response, conversation
    if runner == "gemma4":
        from feed_experiment import install_torchvision_read_video_fallback

        install_torchvision_read_video_fallback()
        # Otherwise Gemma's default 32-frame setting takes precedence over fps.
        prepare_gemma_video_processor(processor)
        # Explicitly supply the soundtrack: video loading alone may discard audio.
        with tempfile.TemporaryDirectory(prefix="gesture_audio_") as folder:
            audio_path = Path(folder) / "soundtrack.wav"
            subprocess.run(["ffmpeg", "-v", "error", "-i", str(video_path), "-vn",
                            "-ac", "1", "-ar", "16000", str(audio_path)], check=True)
            conversation[1]["content"].append({"type": "audio", "audio": str(audio_path)})
            inputs = processor.apply_chat_template(
                conversation, tokenize=True, return_dict=True, return_tensors="pt",
                add_generation_prompt=True, enable_thinking=False,
                processor_kwargs={"videos_kwargs": {"fps": fps, "num_frames": None}},
            ).to(model.device)
            if audit is not None:
                import soundfile as sf

                waveform, sample_rate = sf.read(audio_path)
                audit_inputs(video_path, inputs, audit, audio=waveform, sample_rate=sample_rate)
                audit["processed_video_frames"] = inputs["pixel_values_videos"].shape[1]
                audit["requested_fps"] = fps
        conversation[1]["content"][-1]["audio"] = str(video_path.resolve())
        if not any("audio" in key or "input_features" in key for key in inputs):
            raise RuntimeError("Gemma processor returned no audio features.")
        with torch.inference_mode():
            output = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        audit_generation(inputs, output, max_new_tokens, audit)
        return processor.decode(output[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip(), conversation

    try:
        from qwen_omni_utils import process_mm_info
    except ImportError as exc:
        raise ImportError(
            "Qwen2.5-Omni video loading requires qwen-omni-utils. "
            "Install it with: pip install 'qwen-omni-utils[decord]' -U"
        ) from exc

    text = processor.apply_chat_template(
        conversation,
        add_generation_prompt=True,
        tokenize=False,
    )
    audios, images, videos, video_kwargs = process_mm_info(
        conversation,
        use_audio_in_video=use_audio_in_video,
        return_video_kwargs=True,
        image_patch_size=processor.video_processor.patch_size,
    )
    if use_audio_in_video and (audios is None or len(audios) == 0):
        raise RuntimeError("Video audio is required, but no audio was decoded.")
    inputs = processor(
        text=text,
        audio=audios,
        images=images,
        videos=videos,
        return_tensors="pt",
        padding=True,
        use_audio_in_video=use_audio_in_video,
        videos_kwargs={"fps": video_kwargs["fps"][0], "do_sample_frames": False},
    )
    print(f"Audio-video inputs: {len(audios or [])} audio track(s), "
          f"{len(videos)} video(s), sampled fps={video_kwargs['fps']}")
    if audit is not None:
        audit["decoded_video_shapes"] = [list(video.shape) for video in videos]
        audit["sampled_fps"] = video_kwargs["fps"]
        audit_inputs(video_path, inputs, audit, audio=audios[0],
                     sample_rate=processor.feature_extractor.sampling_rate)
    inputs = inputs.to(model.device).to(model.dtype)

    with torch.inference_mode():
        text_ids = model.generate(
            **inputs,
            thinker_max_new_tokens=max_new_tokens,
            return_audio=False,
            use_audio_in_video=use_audio_in_video,
            thinker_do_sample=False,
        )

    if isinstance(text_ids, tuple):
        text_ids = text_ids[0]
    if hasattr(text_ids, "sequences"):
        text_ids = text_ids.sequences
    audit_generation(inputs, text_ids, max_new_tokens, audit)
    response = processor.batch_decode(
        text_ids[:, inputs.input_ids.shape[1]:],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0].strip()
    return response, conversation


def design_questions(path: Path) -> dict[str, dict[str, str]]:
    """Map Lyn's original gesture design contrasts to video names and questions."""
    questions: dict[str, dict[str, str]] = {}
    targets = {
        "Scalar Implicature": {"X-not-X*"}, "Presupposition": {"p", "everybody-p"},
        "Homogeneity": {"all-Xs", "no-Xs"}, "Supplement": {"if-A-then-supp"},
    }
    with path.open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            family = row["Condition"]
            if family not in targets:
                continue
            story = re.search(r"\d+", row["Group"])[0]
            gesture = row["Gesture.Type"]
            premise = "TargetPremise"
            if family == "Scalar Implicature":
                condition = "Positive" if row["Group"].endswith("a") else "Negative"
                if gesture != ("X" if condition == "Positive" else "not(X*)"):
                    premise = "ControlPremise"
            elif family == "Presupposition":
                condition = "Question" if gesture == "will-pq?" else "None"
            elif family == "Homogeneity":
                condition = "Positive" if gesture == "the-Xs" else "Negative"
            else:
                condition = None
                if gesture != "if-A(supp)-then-C":
                    premise = "ControlPremise"
            stem = f"{family.replace(' ', '')}{story}_{premise}"
            if condition:
                stem += f"({condition})"
            kind = "target" if row["Inference.Type"] in targets[family] else "baseline"
            entry = questions.setdefault(stem, {})
            if kind in entry:
                raise ValueError(f"Duplicate design question: {stem} / {kind}")
            entry[kind] = row["Inference"].strip()
    return questions


def load_jobs(args) -> list[tuple[Path, str, str]]:
    if args.video:
        videos = [args.video]
    else:
        videos = sorted(args.video_dir.glob("*.mp4"))
    materials = {}
    if args.task == "rate":
        if args.design:
            materials = design_questions(args.design)
        else:
            materials = {
                row["task_name"]: {"target": row["target_inference"], "baseline": row["control_inference"]}
                for row in extract_gold(args.stories)
            }
    jobs = []
    for video in videos:
        if video.stem.startswith("Iconicity"):
            print(f"Skipping iconicity trial (different task): {video.name}")
            continue
        if not video.exists():
            raise FileNotFoundError(video)
        if args.task != "rate":
            jobs.append((video, args.task, ""))
            continue
        if video.stem not in materials:
            raise ValueError(f"Missing video or gesture-study statements: {video}")
        for kind in args.statement_types:
            statement = materials[video.stem][kind]
            if not statement:
                raise ValueError(f"Missing {kind} statement: {video.stem}")
            jobs.append((video, kind, statement))
    if not jobs:
        raise ValueError("No inference videos found.")
    return jobs[:args.limit] if args.limit else jobs


def validate_audio(video: Path) -> None:
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "a:0", "-show_entries",
         "stream=codec_name", "-of", "json", str(video)],
        capture_output=True, text=True, check=True,
    )
    if not json.loads(result.stdout).get("streams"):
        raise ValueError(f"No audio track: {video}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rate or describe gesture-study materials using the original video and audio."
    )
    parser.add_argument("--model-id", choices=tuple(MODEL_RUNNERS), default=DEFAULT_MODEL_ID)
    parser.add_argument("--task", choices=("rate", "describe", "sanity"), default="rate")
    parser.add_argument("--audit-inputs", action="store_true", help="Audit modalities and output trimming; automatic for sanity.")
    parser.add_argument("--video", type=Path)
    parser.add_argument("--video-dir", type=Path, default=DEFAULT_VIDEO_DIR)
    parser.add_argument("--stories", type=Path, default=DEFAULT_STORIES)
    parser.add_argument("--design", type=Path, help="Use exact questions from the original human-design CSV instead of the story RTF.")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--limit", type=int, help="Maximum jobs: questions for rate, videos for describe.")
    parser.add_argument("--fps", type=float, default=3.0)
    parser.add_argument("--max-pixels", type=int, default=307200)
    parser.add_argument("--resume", action="store_true", help="Skip previously successful ratings.")
    parser.add_argument(
        "--statement-types",
        nargs="+",
        choices=("target", "baseline"),
        default=["target", "baseline"],
    )
    parser.add_argument("--max-new-tokens", type=int, help="Default: rate=32, describe=2048, sanity=768.")
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
    if args.max_new_tokens is None:
        args.max_new_tokens = {"rate": 32, "describe": 2048, "sanity": 768}[args.task]
    runner = MODEL_RUNNERS[args.model_id]
    if args.fps <= 0 or args.max_pixels <= 0 or args.max_new_tokens <= 0 or (args.limit is not None and args.limit <= 0):
        raise ValueError("Sampling settings and limit must be positive.")
    if (runner == "gemma4" or args.task != "rate") and args.no_audio_in_video:
        raise ValueError("Gemma and gesture perception checks require audio.")
    for field in ("video", "video_dir", "stories", "design", "output"):
        path = getattr(args, field)
        if path is not None and not path.is_absolute():
            setattr(args, field, PROJECT_ROOT / path)
    output_folder, suffix = {
        "rate": ("inference", "ratings"),
        "describe": ("perception/descriptions", "descriptions"),
        "sanity": ("perception/sanity", "sanity"),
    }[args.task]
    args.output = args.output or PROJECT_ROOT / "results" / "gesture" / output_folder / f"{model_slug(args.model_id)}_{suffix}.jsonl"
    if not args.no_audio_in_video and not (
        shutil.which("ffmpeg") or shutil.which("avconv")
    ):
        raise RuntimeError(
            "Audio is enabled because the premise is spoken in the video, but neither "
            "ffmpeg nor avconv is available on PATH. Install/load ffmpeg on the cluster, "
            "or rerun with --no-audio-in-video to ignore the spoken audio."
        )

    jobs = load_jobs(args)
    source_hashes = {video: hashlib.sha256(video.read_bytes()).hexdigest()
                     for video in dict.fromkeys(job[0] for job in jobs)}
    prompt_version = {"rate": RATING_PROMPT_VERSION, "describe": DESCRIPTION_PROMPT_VERSION,
                      "sanity": SANITY_PROMPT_VERSION}[args.task]
    if not args.no_audio_in_video:
        for video in dict.fromkeys(job[0] for job in jobs):
            validate_audio(video)
    print(f"Found {len(jobs)} gesture {args.task} job(s).")
    if args.validate_only:
        for video_path, statement_type, statement in jobs:
            preview = {
                "model": args.model_id,
                "source_video": str(video_path),
                "statement_type": statement_type,
                "statement": statement,
                "conversation": build_conversation(
                    video_path=video_path,
                    statement=statement,
                    task=args.task,
                ),
            }
            print(json.dumps(preview, ensure_ascii=False, indent=2))
        print("Validation passed.")
        return

    import torch
    import transformers

    completed = set()
    if args.output.exists():
        if not args.resume:
            raise FileExistsError(f"Output exists; use --resume or a new --output: {args.output}")
        for line in args.output.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            if row.get("model") != args.model_id or row.get("fps") != args.fps or row.get("max_pixels") != args.max_pixels:
                raise ValueError("Cannot resume results from different model/sampling settings.")
            if runner == "gemma4" and row.get("pipeline_version") != PIPELINE_VERSION:
                raise ValueError("Gemma frame sampling was corrected; use a new --output to avoid mixing old 32-frame results.")
            if row.get("task", "rate") != args.task:
                raise ValueError("Cannot mix inference ratings and perception descriptions.")
            if row.get("prompt_version") != prompt_version or row.get("use_audio_in_video") != (not args.no_audio_in_video):
                raise ValueError("Prompt/audio settings changed; use a new --output.")
            previous_video = Path(row["source_video"])
            if previous_video in source_hashes and row.get("video_sha256") != source_hashes[previous_video]:
                raise ValueError("Video content changed; use a new --output.")
            successful = {
                "describe": row.get("description_json") is not None,
                "rate": row.get("score") is not None,
                # Repetition/budget exhaustion is a model outcome, not a reason to resample.
                "sanity": row.get("sanity_response_present") is True,
            }[args.task]
            if successful and not row.get("error"):
                completed.add((Path(row["source_video"]).name, row["statement_type"], row["statement"]))
    jobs = [job for job in jobs if (job[0].name, job[1], job[2]) not in completed]
    if not jobs:
        print("All requested jobs already completed.")
        return
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; submit this script through job_omni.sh.")
    load_kwargs = {
        "dtype": "auto",
        "device_map": "auto",
        "attn_implementation": "sdpa",
    }
    if args.flash_attention_2:
        load_kwargs["attn_implementation"] = "flash_attention_2"

    print(f"Loading model: {args.model_id}")
    if runner == "minicpmo":
        from omni_runners import load_minicpmo

        model, processor = load_minicpmo(args.model_id)
    else:
        model_class, processor_class = {
            "qwen25_omni": ("Qwen2_5OmniForConditionalGeneration", "Qwen2_5OmniProcessor"),
            "qwen3_omni": ("Qwen3OmniMoeForConditionalGeneration", "Qwen3OmniMoeProcessor"),
            "gemma4": ("AutoModelForMultimodalLM", "AutoProcessor"),
        }[runner]
        model = getattr(transformers, model_class).from_pretrained(args.model_id, **load_kwargs)
        if runner != "gemma4":
            model.disable_talker()
        processor = getattr(transformers, processor_class).from_pretrained(args.model_id)
    print("CUDA available:", torch.cuda.is_available())
    print("Model device:", next(model.parameters()).device)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    failures = 0
    with args.output.open("a" if args.resume else "x", encoding="utf-8") as f:
        for video_path, statement_type, statement in jobs:
            print(f"Running: {video_path.name} [{statement_type}]")
            audit = {} if args.audit_inputs or args.task == "sanity" else None
            try:
                response, conversation = run_one(
                    model=model,
                    processor=processor,
                    video_path=video_path,
                    statement=statement,
                    max_new_tokens=args.max_new_tokens,
                    use_audio_in_video=not args.no_audio_in_video,
                    runner=runner, fps=args.fps, max_pixels=args.max_pixels,
                    task=args.task,
                    audit=audit,
                )
                record = {
                    "model": args.model_id,
                    "runner": runner,
                    "source_video": str(video_path),
                    "statement_type": statement_type,
                    "statement": statement,
                    "response": response,
                    "use_audio_in_video": not args.no_audio_in_video,
                    "conversation": conversation,
                }
                try:
                    if args.task == "describe":
                        record["description_json"] = parse_description(response)
                        timeline = record["description_json"]["timeline"]
                        record["transcript"] = " ".join(
                            event["text"] for event in timeline if event["type"] == "speech"
                        )
                        record["n_gesture_events"] = sum(event["type"] == "gesture" for event in timeline)
                    elif args.task == "sanity":
                        record["sanity_response_present"] = bool(response.strip())
                        if not record["sanity_response_present"]:
                            raise ValueError("Sanity response is empty.")
                    else:
                        cleaned = re.sub(r"^\s*thought\s*\n", "", response, flags=re.IGNORECASE)
                        match = re.fullmatch(r"\s*(\d{1,3})\s*", cleaned)
                        record["score"] = int(match[1]) if match and int(match[1]) <= 100 else None
                        if record["score"] is None:
                            raise ValueError("Response is not a single integer from 0 to 100.")
                except ValueError as exc:
                    record["error"] = str(exc)
                    failures += 1
                print(f"Output preview ({min(500, len(response))}/{len(response)} characters): {response[:500]}")
            except Exception as exc:
                failures += 1
                traceback_text = traceback.format_exc()
                record = {
                    "model": args.model_id,
                    "runner": runner,
                    "source_video": str(video_path),
                    "statement_type": statement_type,
                    "statement": statement,
                    "error": repr(exc),
                    "traceback": traceback_text,
                    "use_audio_in_video": not args.no_audio_in_video,
                }
                print(f"Error on {video_path.name} [{statement_type}]: {exc!r}")
                print(traceback_text)

            record.update(fps=args.fps, max_pixels=args.max_pixels, max_new_tokens=args.max_new_tokens,
                          input_mode="audio-video", task=args.task,
                          pipeline_version=PIPELINE_VERSION,
                          prompt_version=prompt_version, video_sha256=source_hashes[video_path],
                          transformers_version=transformers.__version__, torch_version=torch.__version__)
            if args.task == "rate":
                record["statement_source"] = str(args.design or args.stories)
            if audit is not None:
                record["input_audit"] = audit
                if args.task == "sanity":
                    record["pipeline_checks_passed"] = bool(
                        audit.get("modality_checks_passed") and (audit.get("prompt_prefix_matches") or audit.get("native_answer_only"))
                        and not audit.get("token_budget_reached") and record.get("sanity_response_present")
                        and not record.get("error")
                    )
                print("Input audit:", json.dumps(audit))
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            f.flush()
            print("-" * 80)

    print(f"Wrote {args.task} results to {args.output}")
    if failures:
        raise RuntimeError(f"{failures} jobs failed; inspect the output and rerun with --resume.")


if __name__ == "__main__":
    main()
