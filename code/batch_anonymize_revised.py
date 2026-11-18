#!/usr/bin/env python3
r"""Batch-anonymize gesture videos with per-video adjustments.

Place this file alongside anonymize_videos_auto.py in the code folder.
From the ORIGINAL OSF Stimuli directory, run (Windows CMD):

    python ..\..\..\code\batch_anonymize_revised.py --only-fixes

Outputs go into a NEW sibling folder, 'OSF Stimuli Anonymized Revised',
so your existing anonymized videos will not be replaced. Review the whole
video before using an output as anonymized research material.

To run all input files with the revised per-video rules, omit --only-fixes.
To inspect which source videos/settings will be used: --list-only.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
import subprocess
import sys

EXTENSIONS = {'.mp4', '.mov'}
COMMON_ARGS = [
    '--trace', 'direct', '--blur_radius', '40',
    '--auto-width-scale', '0.55',
]

# Use the source *stem* rather than a suffix, since .mov inputs become .mp4.
# For these five videos, extra height + a slight downward shift protect the chin.
CHIN_FIXES = {
    'supplement1_targetpremise',
    'supplement2_controlpremise',
    'presupposition2_targetpremise(question)',
    'presupposition2_targetpremise(none)',
    'iconicity3_highcrosses',
}

# Two known manually calibrated face boxes. These need shorter/rounder masks.
MANUAL_FIXES = {
    'scalarimplicature4_controlpremise(positive)': (532, 176, 232, 288),
    'scalarimplicature4_targetpremise(positive)': (532, 187, 234, 280),
}

FIX_NAMES = CHIN_FIXES | set(MANUAL_FIXES)


def settings_for(video: Path):
    stem = video.stem.casefold()
    settings = COMMON_ARGS.copy()
    if stem in CHIN_FIXES:
        settings += [
            '--auto-height-scale', '0.80',
            '--auto-y-offset', '-0.02',
        ]
        return 'CHIN FIX: height 0.80, y-offset -0.02', settings
    if stem in MANUAL_FIXES:
        settings += [
            '--auto-height-scale', '0.55',
            '--auto-y-offset', '0.0',
            '--face-box', *(str(v) for v in MANUAL_FIXES[stem]),
        ]
        return f'MANUAL FIX: height 0.55, y-offset 0.0, box {MANUAL_FIXES[stem]}', settings
    settings += ['--auto-height-scale', '0.7', '--auto-y-offset', '-0.10']
    return 'DEFAULT: height 0.70, y-offset -0.10', settings


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--input-dir', type=Path, default=Path.cwd())
    p.add_argument('--output-dir', type=Path, default=None)
    p.add_argument('--only-fixes', action='store_true',
                   help='Process only the seven files with modified settings')
    p.add_argument('--preview-only', action='store_true',
                   help='Generate first-frame ellipse previews only; does NOT assess mid-video tracking')
    p.add_argument('--list-only', action='store_true',
                   help='Print filenames and settings without processing')
    args = p.parse_args()

    source = args.input_dir.resolve()
    destination = (args.output_dir or
                   source.parent / 'OSF Stimuli Anonymized Revised').resolve()
    processor = Path(__file__).resolve().with_name('anonymize_videos_auto.py')
    if not source.is_dir():
        p.error(f'Input directory does not exist: {source}')
    if source == destination:
        p.error('Input and output directories must differ')
    if not processor.is_file() and not args.list_only:
        p.error(f'Expected {processor} alongside this batch script')

    videos = sorted((f for f in source.iterdir()
                     if f.is_file() and f.suffix.lower() in EXTENSIONS
                     and not f.stem.casefold().startswith('test')),
                    key=lambda f: f.name.casefold())
    if args.only_fixes:
        videos = [f for f in videos if f.stem.casefold() in FIX_NAMES]
        found = {f.stem.casefold() for f in videos}
        missing = sorted(FIX_NAMES - found)
        if missing:
            print('WARNING: These expected source names were not found:', flush=True)
            for stem in missing:
                print(f'  - {stem} (.mp4/.mov)', flush=True)
    if not videos:
        p.error('No matching input MP4/MOV videos found')

    outputs = set()
    result_rows = []
    for index, video in enumerate(videos, 1):
        label, settings = settings_for(video)
        target = destination / f'{video.stem}.mp4'
        key = target.name.casefold()
        print(f'[{index}/{len(videos)}] {video.name}\n  {label}', flush=True)

        if key in outputs:
            print('  FAILED: identical output basename from MP4/MOV sources', flush=True)
            result_rows.append([video.name, 'FAILED', label, 'Duplicate source stem'])
            continue
        outputs.add(key)

        if args.list_only:
            result_rows.append([video.name, 'LISTED', label, ''])
            continue

        if target.exists() and not args.preview_only:
            print('  SKIPPED: output exists (no overwrite)', flush=True)
            result_rows.append([video.name, 'SKIPPED', label, 'Output exists'])
            continue

        command = [sys.executable, str(processor), str(video),
                   '--output', str(target), *settings]
        if args.preview_only:
            command += ['--preview-only']
        print('  Running...', flush=True)
        try:
            run = subprocess.run(command)
            status = ('PREVIEWED' if args.preview_only else 'SUCCESS') if run.returncode == 0 else 'FAILED'
            reason = '' if run.returncode == 0 else f'Exit code {run.returncode}'
        except KeyboardInterrupt:
            print('Interrupted; completed files are preserved.', flush=True)
            break
        except OSError as exc:
            status, reason = 'FAILED', str(exc)
        print(f'  {status}: {reason}', flush=True)
        result_rows.append([video.name, status, label, reason])

    if not args.list_only:
        destination.mkdir(parents=True, exist_ok=True)
        report = destination / ('preview_results.csv' if args.preview_only else 'batch_results.csv')
        with report.open('w', encoding='utf-8-sig', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['filename', 'status', 'setting', 'reason'])
            writer.writerows(result_rows)
        print(f'\nOutput folder: {destination}\nReport: {report}', flush=True)
    totals = {status: sum(r[1] == status for r in result_rows)
              for status in ['SUCCESS', 'PREVIEWED', 'SKIPPED', 'FAILED', 'LISTED']}
    print('Summary: ' + ', '.join(f'{n} {s.lower()}' for s, n in totals.items() if n), flush=True)
    if any(r[1] == 'FAILED' for r in result_rows):
        sys.exit(1)


if __name__ == '__main__':
    main()
