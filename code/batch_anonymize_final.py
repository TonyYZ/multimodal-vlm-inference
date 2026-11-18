#!/usr/bin/env python3
r"""Batch anonymize gesture stimuli with two manually calibrated face boxes.

Run from the original OSF Stimuli directory (Windows CMD):
  python ..\..\..\code\batch_anonymize_final.py

Optional first-frame previews (not anonymized; for local quality control only):
  python ..\..\..\code\batch_anonymize_final.py --preview-only

Requires anonymize_videos_auto.py in the same code directory.
"""

import argparse
import csv
from pathlib import Path
import subprocess
import sys

SPECIAL_BOXES = {
    "scalarimplicature4_controlpremise(positive).mov": (532, 176, 232, 288),
    "scalarimplicature4_targetpremise(positive).mov": (532, 187, 234, 280),
}
COMMON_ARGS = [
    "--trace", "direct", "--blur_radius", "40",
    "--auto-width-scale", "0.55", "--auto-height-scale", "0.7",
]
EXTENSIONS = {".mp4", ".mov"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path.cwd(),
                        help="Source folder (default: current working directory)")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Output folder (default: sibling OSF Stimuli Anonymized Final)")
    parser.add_argument("--preview-only", action="store_true",
                        help="Only save unblurred first-frame preview PNGs with red ellipse")
    args = parser.parse_args()

    source = args.input_dir.resolve()
    destination = (args.output_dir or source.parent / "OSF Stimuli Anonymized Final").resolve()
    processor = Path(__file__).resolve().with_name("anonymize_videos_auto.py")

    if not source.is_dir():
        parser.error(f"Source folder does not exist: {source}")
    if not processor.is_file():
        parser.error(f"Cannot find {processor.name} next to this batch script: {processor}")
    if source == destination:
        parser.error("Output directory must differ from input directory")

    videos = sorted((p for p in source.iterdir()
                     if p.is_file() and p.suffix.lower() in EXTENSIONS
                     and not p.stem.lower().startswith("test")),
                    key=lambda p: p.name.casefold())
    if not videos:
        parser.error(f"No MP4/MOV stimulus videos found in {source}")

    folder = destination / "_previews" if args.preview_only else destination
    folder.mkdir(parents=True, exist_ok=True)
    results = []
    targets = set()
    total = len(videos)
    for index, video in enumerate(videos, 1):
        output = folder / f"{video.stem}.mp4"
        canonical = output.name.casefold()
        print(f"\n[{index}/{total}] {video.name}", flush=True)

        if canonical in targets:
            reason = "Output-name collision with another input (.mov/.mp4 share a stem)"
            print(f"FAILED: {reason}", flush=True)
            results.append((video.name, "FAILED", reason))
            continue
        targets.add(canonical)

        if output.exists() and not args.preview_only:
            print("SKIPPED: output already exists (will not overwrite)", flush=True)
            results.append((video.name, "SKIPPED", "Output already exists"))
            continue

        cmd = [sys.executable, str(processor), str(video), "--output", str(output)]
        cmd.extend(COMMON_ARGS)
        box = SPECIAL_BOXES.get(video.name.casefold())
        if box is not None:
            print(f"Manual face box: {box}", flush=True)
            cmd.extend(["--face-box", *(str(v) for v in box)])
        else:
            print("Automatic face detection", flush=True)
        if args.preview_only:
            cmd.append("--preview-only")

        try:
            completed = subprocess.run(cmd, check=False)
        except KeyboardInterrupt:
            print("\nInterrupted. Completed files remain in the output directory.", flush=True)
            break
        except OSError as exc:
            results.append((video.name, "FAILED", str(exc)))
            print(f"FAILED: {exc}", flush=True)
            continue

        status = "PREVIEWED" if args.preview_only else "SUCCESS"
        if completed.returncode == 0:
            results.append((video.name, status, ""))
        else:
            print(f"FAILED: exit code {completed.returncode}", flush=True)
            results.append((video.name, "FAILED", f"Process exit code {completed.returncode}"))

    report = destination / ("preview_results.csv" if args.preview_only else "batch_results.csv")
    with report.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(("filename", "status", "reason"))
        writer.writerows(results)
    successes = sum(status in ("SUCCESS", "PREVIEWED") for _, status, _ in results)
    skipped = sum(status == "SKIPPED" for _, status, _ in results)
    failed = sum(status == "FAILED" for _, status, _ in results)
    print(f"\nCompleted: {successes} successful, {skipped} skipped, {failed} failed")
    print(f"Output folder: {destination}")
    print(f"Results report: {report}")
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
