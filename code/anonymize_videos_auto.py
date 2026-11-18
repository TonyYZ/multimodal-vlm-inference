#!/usr/bin/env python3
"""Anonymize a signer with a video-specific, automatically sized head ellipse.

Requirements: OpenCV contrib with CSRT, Pillow, numpy, ffmpeg and ffprobe.
Works with MP4/MOV and, when given a folder, processes both formats in batch.

Example, from the input folder in Windows CMD:
  python anonymize_videos_auto.py . --output-dir "OSF Stimuli Anonymized Auto" --trace direct --blur_radius 40

Automatic mode uses the included Haar frontal-face XML model to initialize
CSRT. Place haarcascade_frontalface_default.xml beside this script. Detection can fail for side profiles/occlusions. A failed first-frame
face detection *stops processing that clip* instead of using an unsafe default.
Manual rescue for a known video: --face-box X Y WIDTH HEIGHT (source pixels).
Review every output before using it as an anonymized research stimulus.
"""
from __future__ import annotations

import argparse
from fractions import Fraction
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

VIDEO_EXTENSIONS = {'.mp4', '.mov'}


def ffprobe_streams(path: Path) -> list[dict]:
    return json.loads(subprocess.check_output([
        'ffprobe', '-v', 'error', '-show_streams', '-of', 'json', str(path)
    ], text=True))['streams']


def video_info(path: Path) -> tuple[tuple[int, int], str, int | None, str | None]:
    streams = ffprobe_streams(path)
    video = next((x for x in streams if x.get('codec_type') == 'video'), None)
    if video is None:
        raise RuntimeError('No video stream')
    size = int(video['width']), int(video['height'])
    fps = video.get('avg_frame_rate', '0/0')
    if Fraction(fps) <= 0:
        raise RuntimeError('Invalid or unknown video frame rate')
    r_fps = video.get('r_frame_rate', fps)
    if Fraction(r_fps) != Fraction(fps):
        raise RuntimeError(
            f'Likely variable frame rate (r_frame_rate={r_fps}, avg_frame_rate={fps}); '
            'convert to a constant-frame-rate video first to protect audio synchrony'
        )
    declared = video.get('nb_frames')
    expected = int(declared) if declared and declared.isdecimal() else None
    audio = next((x for x in streams if x.get('codec_type') == 'audio'), None)
    return size, fps, expected, audio.get('codec_name') if audio else None


def first_face(frame, face_box: tuple[int, int, int, int] | None):
    import cv2
    height, width = frame.shape[:2]
    if face_box is not None:
        x, y, w, h = face_box
        if min(w, h) <= 0 or x < 0 or y < 0 or x + w > width or y + h > height:
            raise RuntimeError(f'--face-box {face_box} lies outside {width}x{height} video')
        return np.array([x, y, w, h], dtype=float)
    # Python 3.14 / some OpenCV builds omit the cv2.data XML files.
    # Prefer the XML shipped beside this script; otherwise use cv2's bundle.
    filename = 'haarcascade_frontalface_default.xml'
    local_xml = Path(__file__).resolve().parent / filename
    package_xml = Path(getattr(getattr(cv2, 'data', None), 'haarcascades', '')) / filename
    cascade_path = next((p for p in (local_xml, package_xml) if p.is_file()), None)
    if cascade_path is None:
        raise RuntimeError(
            f'Face detector model missing. Put {filename} beside {Path(__file__).name} '
            f'(folder: {Path(__file__).resolve().parent}).'
        )
    detector = cv2.CascadeClassifier(str(cascade_path))
    if detector.empty():
        raise RuntimeError(f'Could not read face detector model: {cascade_path}')
    factor = min(1.0, 960.0 / max(width, height))
    small = cv2.resize(frame, None, fx=factor, fy=factor) if factor < 1 else frame
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    faces = detector.detectMultiScale(gray, scaleFactor=1.08, minNeighbors=5,
                                      minSize=(25, 25))
    if len(faces) == 0:
        raise RuntimeError(
            'No frontal face detected in the first frame. '
            'This video was NOT anonymized. Inspect the first frame and '
            'retry with --face-box X Y W H (full-resolution pixels).'
        )
    if len(faces) > 1:
        print(f'WARNING: Detected {len(faces)} faces; selecting the largest. Review output.',
              flush=True)
    x, y, w, h = max(faces, key=lambda b: b[2] * b[3])
    result = np.array([x, y, w, h], dtype=float) / factor
    return result


def tracker_create(cv2):
    if hasattr(cv2, 'TrackerCSRT_create'):
        return cv2.TrackerCSRT_create()
    if hasattr(cv2, 'legacy') and hasattr(cv2.legacy, 'TrackerCSRT_create'):
        return cv2.legacy.TrackerCSRT_create()
    raise RuntimeError('CSRT not installed. Install ONLY opencv-contrib-python-headless.')


def ellipse_from_face(rect: np.ndarray, width_scale: float, height_scale: float,
                      y_offset: float) -> np.ndarray:
    x, y, w, h = rect
    return np.array([x + w/2, y + h*(0.5 + y_offset),
                     max(2., w*width_scale), max(2., h*height_scale)], dtype=float)


def track_ellipses(video: Path, args, size: tuple[int, int]) -> np.ndarray:
    """Return (cx, cy, radius_x, radius_y), source pixels, for decoded frames."""
    import cv2
    cv2.setNumThreads(1)
    cap = cv2.VideoCapture(str(video))
    try:
        ok, frame = cap.read()
        if not ok:
            raise RuntimeError('OpenCV could not decode first frame')
        face = first_face(frame, args.face_box)
        print('Initial face box (x,y,w,h):', tuple(round(float(v), 1) for v in face), flush=True)
        # Work in a 0.5x frame for efficient CSRT tracking.
        factor = min(1., 640. / max(size))
        frame_small = cv2.resize(frame, None, fx=factor, fy=factor)
        box = tuple(int(round(v * factor)) for v in face)
        box = (box[0], box[1], max(3,box[2]), max(3,box[3]))
        tracker = tracker_create(cv2)
        tracker.init(frame_small, box)
        initial = ellipse_from_face(face, args.auto_width_scale, args.auto_height_scale,
                                    args.auto_y_offset)
        positions = [initial]
        last = initial.copy()
        failed = 0
        consecutive_failures = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            success, bounds = tracker.update(cv2.resize(frame, None, fx=factor, fy=factor))
            if success:
                rect = np.array(bounds, dtype=float) / factor
                # Guard against abrupt failure modes; otherwise mask could jump away.
                candidate = ellipse_from_face(rect, args.auto_width_scale,
                                              args.auto_height_scale, args.auto_y_offset)
                movement = np.linalg.norm(candidate[:2] - last[:2])
                if (movement > args.max_jump or not np.all(np.isfinite(candidate))):
                    success = False
                else:
                    # Relative scale may fluctuate; reject extreme drift from first face.
                    for k in (2, 3):
                        candidate[k] = np.clip(candidate[k], initial[k]*0.7,
                                               initial[k]*1.6)
                    # Smooth only ellipse size, to reduce flicker.
                    candidate[2:] = 0.8 * last[2:] + 0.2 * candidate[2:]
                    last = candidate
                    consecutive_failures = 0
            if not success:
                failed += 1
                consecutive_failures += 1
                if consecutive_failures > args.max_consecutive_failures:
                    raise RuntimeError(
                        f'Face tracker lost subject for {consecutive_failures} '
                        'consecutive frames. Video will not be marked as successful.'
                    )
            positions.append(last.copy())
            if len(positions) % 120 == 0:
                print(f'Tracked {len(positions)} frames', flush=True)
        print(f'Tracking decoded {len(positions)} frames; failures: {failed}', flush=True)
        return np.asarray(positions, dtype=float)
    finally:
        cap.release()


def move_ellipses(samples: np.ndarray, args) -> np.ndarray:
    if args.trace == 'direct':
        return samples
    if args.trace == 'none':
        return np.repeat(samples[:1], len(samples), axis=0)
    out = np.empty_like(samples)
    prev = samples[0].copy()
    for i in range(len(samples)):
        target = samples[min(len(samples)-1, i+args.anticipate)]
        prev = args.inertia*prev + (1-args.inertia)*target
        out[i] = prev
    return out


def blur_ellipse(frame: Image.Image, ellipse: np.ndarray, args) -> Image.Image:
    cx, cy, ax, ay = ellipse
    cx += args.xshift
    cy += args.yshift
    cx, cy, ax, ay = (int(round(float(v))) for v in (cx,cy,ax,ay))
    margin = math.ceil(4*(args.blur_radius + args.feather) + 8)
    bbox = (max(0, cx-ax-margin), max(0, cy-ay-margin),
            min(frame.width, cx+ax+margin+1), min(frame.height, cy+ay+margin+1))
    if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
        raise RuntimeError('Blur ellipse outside frame; failed calibration')
    patch = frame.crop(bbox)
    mask = Image.new('L', patch.size, 0)
    x, y = cx-bbox[0], cy-bbox[1]
    ImageDraw.Draw(mask).ellipse((x-ax, y-ay, x+ax, y+ay), fill=255)
    if args.feather:
        mask = mask.filter(ImageFilter.GaussianBlur(args.feather))
    blurred = patch.filter(ImageFilter.GaussianBlur(args.blur_radius))
    result = frame.copy()
    result.paste(Image.composite(blurred, patch, mask), bbox[:2])
    return result


def read_frame(pipe, bytes_per_frame: int) -> bytes:
    chunks = bytearray()
    while len(chunks) < bytes_per_frame:
        part = pipe.read(bytes_per_frame - len(chunks))
        if not part:
            break
        chunks.extend(part)
    if chunks and len(chunks) != bytes_per_frame:
        raise RuntimeError('FFmpeg decoded incomplete video frame')
    return bytes(chunks)


def preview_first_frame(source: Path, preview_path: Path, args):
    """Save an UNBLURRED first frame with an outline for local calibration QC."""
    import cv2
    cap = cv2.VideoCapture(str(source))
    try:
        ok, bgr = cap.read()
        if not ok:
            raise RuntimeError('Cannot read first frame')
        face = first_face(bgr, args.face_box)
        cx, cy, ax, ay = ellipse_from_face(face, args.auto_width_scale,
                                            args.auto_height_scale, args.auto_y_offset)
        cx += args.xshift
        cy += args.yshift
        img = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img)
        draw.ellipse((round(cx-ax), round(cy-ay), round(cx+ax), round(cy+ay)),
                     outline=(255, 40, 40), width=max(2, img.width // 320))
        preview_path.parent.mkdir(parents=True, exist_ok=True)
        img.save(preview_path)
        print(f'PREVIEW: {preview_path.name} (unblurred original; keep private)', flush=True)
    finally:
        cap.release()


def anonymize(source: Path, output: Path, args) -> str:
    size, fps, declared, audio_codec = video_info(source)
    print(f'\nProcessing: {source.name} ({size[0]}x{size[1]}, fps={fps}, audio={audio_codec})',
          flush=True)
    samples = move_ellipses(track_ellipses(source, args, size), args)
    if declared is not None and declared != len(samples):
        print(f'NOTE: ffprobe declares {declared} frames; OpenCV decoded {len(samples)}. '
              'Will verify against actual FFmpeg frames.', flush=True)

    output.parent.mkdir(parents=True, exist_ok=True)
    temp = output.with_name(f'.{output.stem}.partial.mp4')
    if temp.exists():
        temp.unlink()
    # AAC/MP3 can be copied to MP4. MOV files often hold PCM: convert those to AAC.
    audio_mode = 'copy' if audio_codec in {'aac', 'mp3'} else 'aac'
    writer_cmd = [
        'ffmpeg', '-hide_banner', '-v', 'error', '-y',
        '-f', 'rawvideo', '-pix_fmt', 'rgb24',
        '-s', f'{size[0]}x{size[1]}', '-r', fps, '-i', '-',
        '-i', str(source), '-map', '0:v:0', '-map', '1:a:0?',
        '-c:v', 'libx264', '-preset', 'fast', '-crf', '18',
        '-threads', '2', '-pix_fmt', 'yuv420p',
        '-c:a', audio_mode,
    ]
    if audio_mode == 'aac':
        writer_cmd += ['-b:a', '192k']
    writer_cmd += ['-map_metadata', '-1', '-movflags', '+faststart', str(temp)]
    reader_cmd = ['ffmpeg', '-hide_banner', '-v', 'error', '-i', str(source),
                  '-map', '0:v:0', '-fps_mode', 'passthrough',
                  '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-threads', '1', '-']
    writer = None
    decoder = None
    count = 0
    try:
        writer = subprocess.Popen(writer_cmd, stdin=subprocess.PIPE)
        decoder = subprocess.Popen(reader_cmd, stdout=subprocess.PIPE)
        while True:
            data = read_frame(decoder.stdout, size[0]*size[1]*3)
            if not data:
                break
            if count >= len(samples):
                if count - len(samples) >= 3:
                    raise RuntimeError('FFmpeg decoded >3 more frames than OpenCV; review video')
                ellipse = samples[-1]
            else:
                ellipse = samples[count]
            frame = Image.frombytes('RGB', size, data)
            result = blur_ellipse(frame, ellipse, args)
            writer.stdin.write(result.tobytes())
            count += 1
            if count % 60 == 0:
                print(f'Rendered {count} frames', flush=True)
        if decoder.wait() != 0:
            raise RuntimeError('FFmpeg decoding failed')
        decoder.stdout.close()
        decoder = None
        writer.stdin.close()
        if writer.wait() != 0:
            raise RuntimeError('FFmpeg encoding failed')
        writer = None
        if count < len(samples)-3:
            raise RuntimeError('FFmpeg decoded >3 fewer frames than OpenCV; review video')
        out_size, out_fps, out_count, out_audio_codec = video_info(temp)
        if out_count is None or out_count != count or out_fps != fps:
            raise RuntimeError(f'Output check failed: rendered={count}, '
                               f'output_count={out_count}, output_fps={out_fps}')
        if audio_codec and not out_audio_codec:
            raise RuntimeError('Source has audio, but output has no audio')
        temp.replace(output)
        print(f'OK: {output} ({count} frames, audio={out_audio_codec})', flush=True)
        return 'ok'
    finally:
        if decoder is not None:
            if decoder.stdout:
                decoder.stdout.close()
            if decoder.poll() is None:
                decoder.terminate()
                decoder.wait()
        if writer is not None:
            if writer.stdin and not writer.stdin.closed:
                writer.stdin.close()
            if writer.poll() is None:
                writer.terminate()
                writer.wait()
        if temp.exists():
            temp.unlink()


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('video', type=Path, help='Input MP4/MOV file or directory containing them')
    p.add_argument('--output', type=Path, help='Output MP4 for a single file')
    p.add_argument('--output-dir', type=Path, help='Output folder for directory input')
    p.add_argument('--preview-only', action='store_true',
                   help='Generate first-frame ellipse check PNGs, without encoding videos')
    p.add_argument('--trace', choices=['none', 'direct', 'smooth'], default='direct')
    p.add_argument('--blur_radius', type=float, default=40)
    p.add_argument('--feather', type=float, default=4)
    p.add_argument('--xshift', type=int, default=0, help='Extra manual x correction in source pixels')
    p.add_argument('--yshift', type=int, default=0, help='Extra manual y correction in source pixels')
    p.add_argument('--auto-width-scale', type=float, default=1.20,
                   help='Ellipse half-width divided by detected face box width (default: 1.20)')
    p.add_argument('--auto-height-scale', type=float, default=1.50,
                   help='Ellipse half-height divided by face box height (default: 1.50)')
    p.add_argument('--auto-y-offset', type=float, default=-0.10,
                   help='Ellipse center vertical offset as fraction of face height (negative = up)')
    p.add_argument('--face-box', type=int, nargs=4, metavar=('X','Y','W','H'),
                   help='Manual fallback if Haar cannot see the face in frame 1 (source pixels)')
    p.add_argument('--max-consecutive-failures', type=int, default=12)
    p.add_argument('--max-jump', type=float, default=75,
                   help='Max head-center jump in pixels from one frame to the next')
    p.add_argument('--inertia', type=float, default=0.8)
    p.add_argument('--anticipate', type=int, default=0)
    args = p.parse_args()
    if (min(args.blur_radius, args.feather) < 0 or
        min(args.auto_width_scale, args.auto_height_scale) <= 0 or
        args.max_consecutive_failures < 0 or args.max_jump <= 0 or
        not 0 <= args.inertia < 1 or args.anticipate < 0):
        p.error('Invalid radius, scale, tracker, or smoothing setting')
    if not all(shutil.which(name) for name in ('ffmpeg', 'ffprobe')):
        p.error('ffmpeg and ffprobe must be on PATH')
    if args.video.is_dir():
        if args.output:
            p.error('Use --output-dir (not --output) for folder input')
        if args.face_box:
            p.error('For batch auto-detection, omit --face-box; use per-video mode to rescue failures')
        outdir = args.output_dir or args.video.parent / f'{args.video.name} Anonymized Auto'
        videos = sorted(x for x in args.video.iterdir()
                        if x.is_file() and x.suffix.lower() in VIDEO_EXTENSIONS)
        if not videos:
            p.error('No MP4/MOV videos found in the input folder')
        ok = skipped = failed = 0
        output_names: set[str] = set()
        for src in videos:
            target = outdir / (src.stem + '.mp4')
            collision = target.name.casefold() in output_names
            output_names.add(target.name.casefold())
            if collision:
                print(f'FAILED name collision (.mov/.mp4 share stem): {src.name}', flush=True)
                failed += 1
            elif target.exists() and not args.preview_only:
                print(f'SKIP existing: {target.name}', flush=True)
                skipped += 1
            else:
                try:
                    if args.preview_only:
                        preview_first_frame(src, outdir / '_previews' / (src.stem+'.png'), args)
                    else:
                        anonymize(src, target, args)
                    ok += 1
                except Exception as exc:
                    print(f'FAILED {src.name}: {exc}', flush=True)
                    failed += 1
        print(f'\nBATCH RESULT: {ok} {"previewed" if args.preview_only else "succeeded"}, '
              f'{skipped} skipped, {failed} failed. '
              'Review videos to ensure all faces stay covered.', flush=True)
        if failed:
            sys.exit(1)
    else:
        if not args.video.is_file() or args.video.suffix.lower() not in VIDEO_EXTENSIONS:
            p.error('Input must be an existing MP4/MOV file or directory')
        if args.output_dir:
            p.error('Use --output (not --output-dir) for a single video')
        target = args.output or args.video.with_name(args.video.stem+'_anonymized.mp4')
        if args.preview_only:
            preview_first_frame(args.video, target.with_suffix('.preview.png'), args)
        else:
            if target.exists():
                p.error(f'Output already exists: {target}')
            anonymize(args.video, target, args)


if __name__ == '__main__':
    main()
