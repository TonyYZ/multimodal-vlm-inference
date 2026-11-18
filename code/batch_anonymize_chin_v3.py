#!/usr/bin/env python3
r"""Re-anonymize five chin videos with shorter masks that preserve the lower edge.

Run in Windows CMD from OSF Stimuli:
  python ..\..\..\code\batch_anonymize_chin_v3.py

Optional fine-tuning:
  python ..\..\..\code\batch_anonymize_chin_v3.py --height-scale 0.85 --y-offset 0.13

Keeps other clips and previous anonymized outputs untouched. Saves to a NEW
sibling folder by default. Examine the FULL output clips; a good first-frame
preview alone cannot verify tracking throughout the video.
"""
from pathlib import Path
import argparse
import csv
import subprocess
import sys

TARGETS = {
    'supplement1_targetpremise',
    'supplement2_controlpremise',
    'presupposition2_targetpremise(question)',
    'presupposition2_targetpremise(none)',
    'iconicity3_highcrosses',
}
EXTENSIONS = {'.mp4', '.mov'}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--input-dir', type=Path, default=Path.cwd())
    parser.add_argument('--output-dir', type=Path, default=None)
    parser.add_argument('--height-scale', type=float, default=0.80)
    parser.add_argument('--y-offset', type=float, default=0.18)
    parser.add_argument('--list-only', action='store_true')
    args = parser.parse_args()
    if args.height_scale <= 0:
        parser.error('--height-scale must be positive')
    source = args.input_dir.resolve()
    if not source.is_dir():
        parser.error(f'Input directory not found: {source}')
    output_dir = (args.output_dir or source.parent / 'OSF Stimuli Anonymized Chin V3').resolve()
    if source == output_dir:
        parser.error('Source and output directory cannot be the same')
    processor = Path(__file__).resolve().with_name('anonymize_videos_auto.py')
    if not processor.is_file() and not args.list_only:
        parser.error(f'Missing anonymizer: {processor}')
    matched = sorted((p for p in source.iterdir()
                      if p.is_file() and p.suffix.casefold() in EXTENSIONS
                      and p.stem.casefold() in TARGETS), key=lambda p: p.name.casefold())
    present = {p.stem.casefold() for p in matched}
    missing = TARGETS - present
    if missing:
        print('WARNING missing originals:', ', '.join(sorted(missing)), flush=True)
    outputs = set()
    rows = []
    for index, video in enumerate(matched, start=1):
        output = output_dir / f'{video.stem}.mp4'
        duplicate = output.name.casefold() in outputs
        outputs.add(output.name.casefold())
        if duplicate:
            status = 'FAILED: duplicate .mov/.mp4 stem'
        elif args.list_only:
            status = 'LISTED'
        elif output.exists():
            status = 'SKIPPED: output already exists'
        else:
            print(f'[{index}/{len(matched)}] Processing {video.name}', flush=True)
            command = [
                sys.executable, str(processor), str(video), '--output', str(output),
                '--trace', 'direct', '--blur_radius', '40',
                '--auto-width-scale', '0.65' if video.stem.casefold() == 'presupposition2_targetpremise(question)' else '0.55',
                '--auto-height-scale', str(args.height_scale),
                '--auto-y-offset', str(args.y_offset),
            ]
            if video.stem.casefold() == 'presupposition2_targetpremise(question)':
                command.extend(['--xshift', '-10'])
            try:
                result = subprocess.run(command)
                status = 'SUCCESS' if result.returncode == 0 else f'FAILED: exit code {result.returncode}'
            except OSError as exc:
                status = f'FAILED: {exc}'
        print(f'  {video.name}: {status}', flush=True)
        rows.append((video.name, status, f'{args.height_scale}', f'{args.y_offset}'))
    if not args.list_only:
        output_dir.mkdir(parents=True, exist_ok=True)
        with (output_dir / 'batch_results.csv').open('w', newline='', encoding='utf-8-sig') as f:
            writer = csv.writer(f)
            writer.writerow(('filename', 'status', 'height_scale', 'y_offset'))
            writer.writerows(rows)
        print(f'\nOutput: {output_dir}', flush=True)
    if any(row[1].startswith('FAILED') for row in rows):
        sys.exit(1)

if __name__ == '__main__':
    main()
