from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import av
import numpy as np
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_VIDEO_DIR = PROJECT_ROOT / "materials" / "animations"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "processed" / "split_animations"

VIDEO_EXTENSIONS = {".mp4", ".m4v", ".mov", ".webm", ".avi"}


@dataclass
class SampledFrame:
    time: float
    image: Image.Image
    is_animation: bool
    has_text_like_content: bool
    shape_score: int
    color_score: int


@dataclass
class Span:
    start: float
    end: float
    kind: str
    frame_indices: list[int]

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)


def iter_video_paths(video_dir: Path) -> list[Path]:
    return sorted(path for path in video_dir.iterdir() if path.suffix.lower() in VIDEO_EXTENSIONS)


def video_duration(video_path: Path) -> float:
    with av.open(str(video_path)) as container:
        if container.duration is None:
            stream = container.streams.video[0]
            if stream.duration is None or stream.time_base is None:
                raise ValueError(f"Cannot determine duration for {video_path}")
            return float(stream.duration * stream.time_base)
        return float(container.duration / av.time_base)


def sample_video(video_path: Path, sample_fps: float) -> list[tuple[float, Image.Image]]:
    step = 1.0 / sample_fps
    next_time = 0.0
    frames: list[tuple[float, Image.Image]] = []

    with av.open(str(video_path)) as container:
        stream = container.streams.video[0]
        for frame in container.decode(stream):
            timestamp = frame.time
            if timestamp is None:
                continue
            if timestamp + 1e-6 < next_time:
                continue
            image = frame.to_image().convert("RGB")
            frames.append((float(timestamp), image))
            next_time += step

    return frames


def downsample_for_analysis(image: Image.Image, width: int = 320) -> np.ndarray:
    ratio = width / image.width
    height = max(1, int(round(image.height * ratio)))
    return np.asarray(image.resize((width, height), Image.Resampling.BILINEAR))


def connected_components(mask: np.ndarray) -> list[tuple[int, int, int, int, int]]:
    height, width = mask.shape
    visited = np.zeros(mask.shape, dtype=bool)
    components: list[tuple[int, int, int, int, int]] = []

    for y in range(height):
        for x in range(width):
            if not mask[y, x] or visited[y, x]:
                continue

            stack = [(x, y)]
            visited[y, x] = True
            area = 0
            min_x = max_x = x
            min_y = max_y = y

            while stack:
                cx, cy = stack.pop()
                area += 1
                min_x = min(min_x, cx)
                max_x = max(max_x, cx)
                min_y = min(min_y, cy)
                max_y = max(max_y, cy)

                for ny in range(max(0, cy - 1), min(height, cy + 2)):
                    for nx in range(max(0, cx - 1), min(width, cx + 2)):
                        if not visited[ny, nx] and mask[ny, nx]:
                            visited[ny, nx] = True
                            stack.append((nx, ny))

            components.append((area, min_x, min_y, max_x, max_y))

    return components


def classify_frame(
    image: Image.Image,
    ink_threshold: int,
    color_delta_threshold: int,
    min_shape_area: int,
    min_shape_extent: int,
) -> tuple[bool, bool, int, int]:
    arr = downsample_for_analysis(image)
    channel_min = arr.min(axis=2)
    channel_max = arr.max(axis=2)

    ink_mask = channel_min < ink_threshold
    color_mask = (channel_max - channel_min > color_delta_threshold) & ink_mask

    # Ignore the occasional near-black full-screen end frame: it is a transition,
    # not useful text or shape content for the downstream VLM.
    mostly_dark = float(channel_max.mean()) < 40.0
    if mostly_dark:
        return False, False, 0, 0

    ink_score = int(ink_mask.sum())
    if ink_score == 0:
        return False, False, 0, 0

    ys, xs = np.nonzero(ink_mask)
    ink_width = int(xs.max() - xs.min() + 1)
    ink_height = int(ys.max() - ys.min() + 1)
    ink_aspect = ink_width / max(1, ink_height)
    looks_like_text_screen = ink_aspect > 4.0 and ink_height <= arr.shape[0] * 0.16

    components = connected_components(ink_mask)
    shape_score = 0
    text_like_score = 0
    text_like_components = 0
    word_like_score = 0
    word_like_components = 0

    for area, min_x, min_y, max_x, max_y in components:
        width = max_x - min_x + 1
        height = max_y - min_y + 1
        aspect = width / max(1, height)

        is_text_line = aspect > 4.5 and height <= min_shape_extent * 2
        is_word_like = height <= min_shape_extent * 3 and area <= min_shape_area * 20
        is_shape_like = area >= min_shape_area and width >= min_shape_extent and height >= min_shape_extent

        if is_text_line:
            text_like_score += area
            text_like_components += 1
        elif is_word_like:
            word_like_score += area
            word_like_components += 1
        elif is_shape_like:
            shape_score += area

    color_score = int(color_mask.sum())
    looks_like_multiline_text = (
        (text_like_components >= 2 and text_like_score >= ink_score * 0.45)
        or (word_like_components >= 6 and word_like_score >= ink_score * 0.45)
        and color_score < min_shape_area
    )
    looks_like_text_screen = looks_like_text_screen or looks_like_multiline_text
    is_animation = (shape_score > 0 or color_score >= min_shape_area) and not looks_like_text_screen
    has_text_like_content = ink_score > 20 and not is_animation

    return is_animation, has_text_like_content, shape_score, color_score


def merge_indices(
    samples: list[SampledFrame],
    wanted: list[bool],
    kind: str,
    sample_step: float,
    gap_tolerance: float,
    min_duration: float,
    start_padding: float,
    end_padding: float,
    duration: float,
    split_change_threshold: float | None = None,
) -> list[Span]:
    spans: list[Span] = []
    current: list[int] = []
    last_time: float | None = None

    for index, (sample, keep) in enumerate(zip(samples, wanted)):
        if not keep:
            continue

        changed = False
        if current and split_change_threshold is not None:
            changed = image_difference(samples[current[-1]].image, sample.image) > split_change_threshold

        if current and last_time is not None and (sample.time - last_time > gap_tolerance or changed):
            spans.append(make_span(samples, current, kind, sample_step, start_padding, end_padding, duration))
            current = []

        current.append(index)
        last_time = sample.time

    if current:
        spans.append(make_span(samples, current, kind, sample_step, start_padding, end_padding, duration))

    return [span for span in spans if span.duration >= min_duration]


def image_difference(left: Image.Image, right: Image.Image, width: int = 160) -> float:
    def prepare(image: Image.Image) -> np.ndarray:
        ratio = width / image.width
        height = max(1, int(round(image.height * ratio)))
        resized = image.resize((width, height), Image.Resampling.BILINEAR).convert("L")
        return np.asarray(resized, dtype=np.float32)

    left_arr = prepare(left)
    right_arr = prepare(right)
    return float(np.abs(left_arr - right_arr).mean())


def make_span(
    samples: list[SampledFrame],
    indices: list[int],
    kind: str,
    sample_step: float,
    start_padding: float,
    end_padding: float,
    duration: float,
) -> Span:
    start = max(0.0, samples[indices[0]].time - sample_step / 2.0 - start_padding)
    end = min(duration, samples[indices[-1]].time + sample_step / 2.0 + end_padding)
    return Span(start=start, end=end, kind=kind, frame_indices=indices)


def merge_close_spans(spans: list[Span], max_gap: float) -> list[Span]:
    if not spans:
        return []

    merged = [spans[0]]
    for span in spans[1:]:
        previous = merged[-1]
        if span.start - previous.end <= max_gap:
            previous.end = max(previous.end, span.end)
            previous.frame_indices.extend(span.frame_indices)
        else:
            merged.append(span)
    return merged


def clamp_animation_spans_after_text(
    animation_spans: list[Span],
    text_spans: list[Span],
    safety_margin: float,
) -> list[Span]:
    for animation_span in animation_spans:
        for text_span in text_spans:
            text_boundary = text_span.end + safety_margin
            too_close = text_span.start < animation_span.end and text_boundary > animation_span.start
            if too_close and text_boundary < animation_span.end:
                animation_span.start = max(animation_span.start, text_boundary)
    return [span for span in animation_spans if span.end > span.start]


def align_animation_starts_to_first_sample(animation_spans: list[Span], samples: list[SampledFrame]) -> list[Span]:
    for span in animation_spans:
        if span.frame_indices:
            span.start = max(span.start, samples[span.frame_indices[0]].time)
    return [span for span in animation_spans if span.end > span.start]


def remove_duplicate_text_spans(
    text_spans: list[Span],
    samples: list[SampledFrame],
    duplicate_threshold: float,
) -> list[Span]:
    kept: list[Span] = []
    kept_images: list[Image.Image] = []

    for span in text_spans:
        image = representative_frame(samples, span)
        is_duplicate = any(image_difference(image, kept_image) <= duplicate_threshold for kept_image in kept_images)
        if not is_duplicate:
            kept.append(span)
            kept_images.append(image)

    return kept


def representative_frame(samples: list[SampledFrame], span: Span) -> Image.Image:
    midpoint = (span.start + span.end) / 2.0
    best_index = min(span.frame_indices, key=lambda idx: abs(samples[idx].time - midpoint))
    return samples[best_index].image


def clip_video(source: Path, target: Path, start: float, duration: float) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(source),
        "-ss",
        f"{start:.3f}",
        "-t",
        f"{duration:.3f}",
        "-an",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(target),
    ]
    subprocess.run(command, check=True)


def process_video(args: argparse.Namespace, video_path: Path) -> dict:
    duration = video_duration(video_path)
    raw_samples = sample_video(video_path, args.sample_fps)
    samples: list[SampledFrame] = []

    for time, image in raw_samples:
        is_animation, has_text, shape_score, color_score = classify_frame(
            image=image,
            ink_threshold=args.ink_threshold,
            color_delta_threshold=args.color_delta_threshold,
            min_shape_area=args.min_shape_area,
            min_shape_extent=args.min_shape_extent,
        )
        samples.append(
            SampledFrame(
                time=time,
                image=image,
                is_animation=is_animation,
                has_text_like_content=has_text,
                shape_score=shape_score,
                color_score=color_score,
            )
        )

    sample_step = 1.0 / args.sample_fps
    animation_spans = merge_indices(
        samples=samples,
        wanted=[sample.is_animation for sample in samples],
        kind="animation",
        sample_step=sample_step,
        gap_tolerance=args.gap_tolerance,
        min_duration=args.min_animation_duration,
        start_padding=args.animation_start_padding,
        end_padding=args.animation_end_padding,
        duration=duration,
    )
    animation_spans = merge_close_spans(animation_spans, args.merge_close_animation_gap)
    text_spans = merge_indices(
        samples=samples,
        wanted=[sample.has_text_like_content for sample in samples],
        kind="text",
        sample_step=sample_step,
        gap_tolerance=args.gap_tolerance,
        min_duration=args.min_text_duration,
        start_padding=0.0,
        end_padding=0.0,
        duration=duration,
        split_change_threshold=args.text_change_threshold,
    )
    animation_spans = clamp_animation_spans_after_text(
        animation_spans,
        text_spans,
        args.text_animation_safety_margin,
    )
    animation_spans = align_animation_starts_to_first_sample(animation_spans, samples)
    text_spans = remove_duplicate_text_spans(
        text_spans,
        samples,
        args.duplicate_text_threshold,
    )

    video_out_dir = args.output_dir / video_path.stem
    if video_out_dir.exists():
        shutil.rmtree(video_out_dir)

    text_frame_dir = video_out_dir / "text_frames"
    animation_clip_dir = video_out_dir / "animation_clips"
    text_frame_dir.mkdir(parents=True, exist_ok=True)
    animation_clip_dir.mkdir(parents=True, exist_ok=True)

    text_records = []
    for span_index, span in enumerate(text_spans):
        frame_path = text_frame_dir / f"text_{span_index:03d}.jpg"
        representative_frame(samples, span).save(frame_path, quality=95)
        text_records.append(
            {
                "index": span_index,
                "start": round(span.start, 3),
                "end": round(span.end, 3),
                "duration": round(span.duration, 3),
                "representative_frame": str(frame_path.relative_to(PROJECT_ROOT)),
                "transcription": "",
            }
        )

    animation_records = []
    for span_index, span in enumerate(animation_spans):
        clip_path = animation_clip_dir / f"animation_{span_index:03d}.mp4"
        if not args.no_clip:
            clip_video(video_path, clip_path, span.start, span.duration)
        animation_records.append(
            {
                "index": span_index,
                "start": round(span.start, 3),
                "end": round(span.end, 3),
                "duration": round(span.duration, 3),
                "clip": str(clip_path.relative_to(PROJECT_ROOT)),
            }
        )

    manifest = {
        "source_video": str(video_path.relative_to(PROJECT_ROOT)),
        "duration": round(duration, 3),
        "sample_fps": args.sample_fps,
        "text_spans": text_records,
        "animation_spans": animation_records,
        "debug_samples": [
            {
                "time": round(sample.time, 3),
                "is_animation": sample.is_animation,
                "has_text_like_content": sample.has_text_like_content,
                "shape_score": sample.shape_score,
                "color_score": sample.color_score,
            }
            for sample in samples
        ],
    }

    manifest_path = video_out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Split animation videos into text-screen frames and geometric-animation clips."
    )
    parser.add_argument("--video-dir", type=Path, default=DEFAULT_VIDEO_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--sample-fps", type=float, default=4.0)
    parser.add_argument("--ink-threshold", type=int, default=235)
    parser.add_argument("--color-delta-threshold", type=int, default=35)
    parser.add_argument("--min-shape-area", type=int, default=35)
    parser.add_argument("--min-shape-extent", type=int, default=9)
    parser.add_argument("--gap-tolerance", type=float, default=0.55)
    parser.add_argument("--merge-close-animation-gap", type=float, default=0.75)
    parser.add_argument("--animation-start-padding", type=float, default=0.0)
    parser.add_argument("--animation-end-padding", type=float, default=0.2)
    parser.add_argument("--text-animation-safety-margin", type=float, default=0.08)
    parser.add_argument("--min-animation-duration", type=float, default=0.25)
    parser.add_argument("--min-text-duration", type=float, default=0.5)
    parser.add_argument("--text-change-threshold", type=float, default=1.5)
    parser.add_argument("--duplicate-text-threshold", type=float, default=0.5)
    parser.add_argument("--no-clip", action="store_true", help="Write manifests and text frames without ffmpeg clips.")
    parser.add_argument("--limit", type=int, default=None, help="Process only the first N videos.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.video_dir = args.video_dir.resolve()
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    video_paths = iter_video_paths(args.video_dir)
    if args.limit is not None:
        video_paths = video_paths[: args.limit]
    if not video_paths:
        raise FileNotFoundError(f"No video files found in {args.video_dir}")

    all_manifests = []
    for video_path in video_paths:
        print(f"Splitting: {video_path.name}")
        manifest = process_video(args, video_path)
        all_manifests.append(manifest)
        print(
            f"  text spans: {len(manifest['text_spans'])}, "
            f"animation spans: {len(manifest['animation_spans'])}"
        )

    index_path = args.output_dir / "index.jsonl"
    with index_path.open("w", encoding="utf-8") as f:
        for manifest in all_manifests:
            f.write(json.dumps(manifest, ensure_ascii=False) + "\n")

    print(f"Wrote index: {index_path}")


if __name__ == "__main__":
    main()
