from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import torch
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration


MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SPLIT_DIR = PROJECT_ROOT / "processed" / "split_animations"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "processed" / "vlm_sequences"
DEFAULT_STORY_FILE = PROJECT_ROOT / "materials" / "stories" / "TSC_EXP_2_Stories.rtf"
TEXT_REPLACEMENTS = {
    "\u2018": "'",
    "\u2019": "'",
    "\u201c": '"',
    "\u201d": '"',
    "\u2026": "...",
}


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def relative_or_absolute(path: Path, absolute: bool) -> str:
    if absolute:
        return str(path.resolve())
    return str(path.relative_to(PROJECT_ROOT))


def image_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rtf_to_text(path: Path) -> str:
    raw = path.read_text(encoding="cp1252")

    def decode_hex(match: re.Match[str]) -> str:
        return bytes.fromhex(match.group(1)).decode("cp1252")

    text = re.sub(r"\\'([0-9a-fA-F]{2})", decode_hex, raw)
    text = text.replace("\\\n", "\n")
    text = re.sub(r"\\[a-zA-Z]+-?\d* ?", "", text)
    text = text.replace(r"\{", "{").replace(r"\}", "}").replace(r"\\", "\\")
    text = text.replace("{", "").replace("}", "").replace("\\", "")
    return text


def clean_story_line(line: str) -> str:
    line = line.strip()
    for old, new in TEXT_REPLACEMENTS.items():
        line = line.replace(old, new)
    line = re.sub(r"^Slide\s+\d+\s*:\s*", "", line)
    line = line.replace("[animation]", "Animation:")
    line = line.replace("*", "")
    return re.sub(r"\s+", " ", line).strip()


def load_story_materials(path: Path) -> tuple[dict[str, str], dict[str, list[str]]]:
    if not path.exists():
        raise FileNotFoundError(f"Cannot find story file: {path}")

    heading_pattern = re.compile(
        r"^(?:Homogeneity|Presupposition|ScalarImplicature|Supplement)\d+_"
        r"(?:TargetPremise|ControlPremise)(?:\([^)]+\))?$"
    )
    stories: dict[str, str] = {}
    text_slides: dict[str, list[str]] = {}
    current_task: str | None = None
    current_lines: list[str] = []
    current_text_slides: list[str] = []
    in_inferences = False

    def save_current() -> None:
        if current_task is None:
            return
        source_video = f"materials\\animations\\{current_task}.m4v"
        story = "\n".join(line for line in current_lines if line).strip()
        if story:
            stories[source_video] = story
        if current_text_slides:
            text_slides[source_video] = list(current_text_slides)

    for raw_line in rtf_to_text(path).splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if heading_pattern.match(line):
            save_current()
            current_task = line
            current_lines = []
            current_text_slides = []
            in_inferences = False
            continue
        if current_task is None:
            continue
        if re.match(r"^(Target|Control) inference\s*:", line):
            in_inferences = True
            continue
        if in_inferences:
            continue
        cleaned = clean_story_line(line)
        if re.match(r"^(Target|Control) inference\s*:", cleaned):
            in_inferences = True
            continue
        if cleaned:
            current_lines.append(cleaned)
            if not cleaned.startswith("Animation:"):
                current_text_slides.append(cleaned)

    save_current()
    return stories, text_slides


def load_qwen_model() -> tuple[Qwen2_5_VLForConditionalGeneration, AutoProcessor]:
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_ID,
        torch_dtype="auto",
        device_map="auto",
    )
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    return model, processor


def transcribe_frame(
    model: Qwen2_5_VLForConditionalGeneration,
    processor: AutoProcessor,
    image_path: Path,
    max_new_tokens: int,
) -> str:
    prompt = (
        "Transcribe exactly the visible text in this image. "
        "Return only the text, preserving punctuation and ellipses. "
        "Do not describe the image."
    )
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": str(image_path.resolve())},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
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


def fill_transcriptions(
    manifests: list[tuple[Path, dict]],
    max_new_tokens: int,
    overwrite: bool,
) -> None:
    model, processor = load_qwen_model()
    cache: dict[str, str] = {}

    for manifest_path, manifest in manifests:
        changed = False
        for text_span in manifest.get("text_spans", []):
            if text_span.get("transcription") and not overwrite:
                continue
            frame_path = PROJECT_ROOT / text_span["representative_frame"]
            key = image_hash(frame_path)
            if key not in cache:
                print(f"Transcribing: {frame_path.relative_to(PROJECT_ROOT)}")
                cache[key] = transcribe_frame(model, processor, frame_path, max_new_tokens)
            text_span["transcription"] = cache[key]
            changed = True

        if changed:
            write_json(manifest_path, manifest)


def build_sequence(
    manifest: dict,
    video_fps: float,
    max_pixels: int,
    absolute_paths: bool,
    include_empty_text: bool,
    story_texts: dict[str, str],
    story_text_slides: dict[str, list[str]],
) -> dict:
    items = []
    fallback_texts = story_text_slides.get(manifest["source_video"], [])

    for text_span in manifest.get("text_spans", []):
        transcription = text_span.get("transcription", "").strip()
        if not transcription:
            text_index = int(text_span.get("index", len(items)))
            if text_index < len(fallback_texts):
                transcription = fallback_texts[text_index]
        if transcription or include_empty_text:
            items.append(
                {
                    "type": "text",
                    "start": text_span["start"],
                    "end": text_span["end"],
                    "text": transcription,
                    "source_frame": text_span["representative_frame"],
                }
            )

    for animation_span in manifest.get("animation_spans", []):
        clip_path = PROJECT_ROOT / animation_span["clip"]
        items.append(
            {
                "type": "video",
                "start": animation_span["start"],
                "end": animation_span["end"],
                "video": relative_or_absolute(clip_path, absolute_paths),
                "fps": video_fps,
                "max_pixels": max_pixels,
            }
        )

    items.sort(key=lambda item: (item["start"], 0 if item["type"] == "text" else 1))

    qwen_content = []
    for item in items:
        if item["type"] == "text":
            if item["text"]:
                qwen_content.append({"type": "text", "text": item["text"]})
        else:
            qwen_content.append(
                {
                    "type": "video",
                    "video": item["video"],
                    "fps": item["fps"],
                    "max_pixels": item["max_pixels"],
                }
            )

    source_video_path = PROJECT_ROOT / manifest["source_video"]
    qwen_content_pure_video = [
        {
            "type": "video",
            "video": relative_or_absolute(source_video_path, absolute_paths),
            "fps": video_fps,
            "max_pixels": max_pixels,
        }
    ]

    story_text = story_texts.get(manifest["source_video"], "")
    qwen_content_pure_text = [{"type": "text", "text": story_text}] if story_text else []

    return {
        "source_video": manifest["source_video"],
        "items": items,
        "qwen_content": qwen_content,
        "qwen_content_pure_video": qwen_content_pure_video,
        "qwen_content_pure_text": qwen_content_pure_text,
    }


def manifest_paths(split_dir: Path) -> list[Path]:
    return sorted(path for path in split_dir.glob("*/manifest.json") if path.is_file())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build VLM-ready text/video sequences from split animation manifests."
    )
    parser.add_argument("--split-dir", type=Path, default=DEFAULT_SPLIT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--story-file", type=Path, default=DEFAULT_STORY_FILE)
    parser.add_argument("--transcribe-with-qwen", action="store_true")
    parser.add_argument("--overwrite-transcriptions", action="store_true")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--video-fps", type=float, default=3.0)
    parser.add_argument("--max-pixels", type=int, default=640 * 480)
    parser.add_argument("--absolute-paths", action="store_true")
    parser.add_argument("--include-empty-text", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    split_dir = args.split_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    paths = manifest_paths(split_dir)
    if args.limit is not None:
        paths = paths[: args.limit]
    if not paths:
        raise FileNotFoundError(f"No manifest files found in {split_dir}")

    manifests = [(path, load_json(path)) for path in paths]
    story_texts, story_text_slides = load_story_materials(args.story_file)

    if args.transcribe_with_qwen:
        fill_transcriptions(
            manifests,
            max_new_tokens=args.max_new_tokens,
            overwrite=args.overwrite_transcriptions,
        )
        manifests = [(path, load_json(path)) for path, _ in manifests]

    sequence_records = []
    for manifest_path, manifest in manifests:
        sequence = build_sequence(
            manifest,
            video_fps=args.video_fps,
            max_pixels=args.max_pixels,
            absolute_paths=args.absolute_paths,
            include_empty_text=args.include_empty_text,
            story_texts=story_texts,
            story_text_slides=story_text_slides,
        )
        video_out_dir = output_dir / Path(manifest["source_video"]).stem
        sequence_path = video_out_dir / "sequence.json"
        write_json(sequence_path, sequence)
        sequence_records.append(sequence)
        print(
            f"Built: {sequence_path.relative_to(PROJECT_ROOT)} "
            f"({len(sequence['items'])} ordered items)"
        )

    index_path = output_dir / "index.jsonl"
    with index_path.open("w", encoding="utf-8") as f:
        for sequence in sequence_records:
            f.write(json.dumps(sequence, ensure_ascii=False) + "\n")
    print(f"Wrote index: {index_path}")


if __name__ == "__main__":
    main()
