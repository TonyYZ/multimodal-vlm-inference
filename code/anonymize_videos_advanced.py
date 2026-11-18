#!/usr/bin/env python3
"""Compare or render face blurring; this standalone file contains every setting.

INSTALL (Python 3.9+):
  python -m pip install Pillow==11.3.0 numpy==2.0.2 opencv-contrib-python-headless==4.10.0.84
Also install FFmpeg/ffprobe with libx264 (macOS: brew install ffmpeg).

Put the original video beside this script and run from that folder:
  python anonymize_videos.py 'Homogeneity1_TargetPremise(Positive).mp4' --grid
  python anonymize_videos.py 'Homogeneity1_TargetPremise(Positive).mp4' --trace smooth --inertia 0.8 --anticipate 3
  python anonymize_videos.py --help

Individual commands beneath grid panels create anonymized.mp4. Use --output
for another filename; existing files are never overwritten. The required
video argument selects the source. Grid output is anonymization_grid.mp4. All outputs are silent.
The grid includes the unblurred original: it is a review video.

TRACE MODES:
  none: fixed ellipse.
  direct: CSRT tracks the face on every frame; ellipse follows its displacement.
  smooth: inertia filters that same trajectory. Higher inertia = slower motion.
          --anticipate N uses the tracked position N frames ahead as its target
          to offset lag (offline processing, not speculative prediction).
On tracking failure the last valid location is retained, never an unblurred
frame; failures are reported. Masks do not change size when the tracker does.

Coordinates are pixels in the source, origin top-left. Default center (710,255).
--xshift/--yshift offset this center; --axes gives ellipse HALF-width/height.
--blur_radius and --feather are Pillow GaussianBlur radii. A large/shifted mask
can obscure hands near the face. Masks and tracking initialization are tuned
for this 1280x720 clip; review/recalibrate them for other material. Familiar
viewers may still recognize the signer. The blur/mask operations follow the
Pillow helpers in the original anonymize.py. No additional files are needed.

Video: all source frames, including first/last; no introductory still, freeze,
trim or speed change. H.264 CRF18, yuv420p, source constant frame rate.
"""
import argparse
from fractions import Fraction
import json
import math
from pathlib import Path
import shutil
import shlex
import subprocess

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

SOURCE = 'Homogeneity1_TargetPremise(Positive).mp4'
DEFAULTS = dict(trace='none', xshift=0, yshift=0, axes=(105,135),
                blur_radius=24, feather=4, inertia=0.8, anticipate=0)
# Each entry is an ordinary set of CLI parameters, not a hidden preset.
EXAMPLES = [
 ('1 Fixed / original-sized mask', dict(blur_radius=12,feather=0)),
 ('2 Direct / original-sized mask', dict(trace='direct',blur_radius=12,feather=0)),
 ('3 Inertia / same mask', dict(trace='smooth',blur_radius=12,feather=0,inertia=0.8)),
 ('4 Inertia + anticipation', dict(trace='smooth',blur_radius=12,feather=0,inertia=0.8,anticipate=4)),
 ('5 Fixed / small face mask', dict(axes=(85,105),blur_radius=18)),
 ('6 Direct / small face mask', dict(trace='direct',axes=(85,105),blur_radius=18)),
 ('7 Gentle inertia / face mask', dict(trace='smooth',axes=(90,110),blur_radius=24,inertia=0.5)),
 ('8 Heavy inertia / face mask', dict(trace='smooth',axes=(90,110),blur_radius=24,inertia=0.9,anticipate=6)),
 ('9 Fixed / stronger blur', dict(blur_radius=40)),
 ('10 Direct / stronger blur', dict(trace='direct',blur_radius=40)),
 ('11 Inertia / soft edge', dict(trace='smooth',blur_radius=24,feather=16,inertia=0.8,anticipate=3)),
 ('12 Fixed / very soft edge', dict(axes=(120,150),blur_radius=32,feather=24)),
 ('13 Fixed / shifted left', dict(xshift=-18,yshift=0,blur_radius=24)),
 ('14 Direct / shifted right', dict(trace='direct',xshift=18,blur_radius=24)),
 ('15 Inertia / shifted upward', dict(trace='smooth',yshift=-18,axes=(112,145),blur_radius=24,inertia=0.7,anticipate=2)),
 ('16 Fixed / shifted downward', dict(yshift=20,axes=(112,145),blur_radius=32)),
 ('17 Fixed / broad head and neck', dict(axes=(135,175),blur_radius=24,feather=8)),
 ('18 Direct / broad head and neck', dict(trace='direct',axes=(135,175),blur_radius=32,feather=8)),
 ('19 Inertia / broad head and neck', dict(trace='smooth',axes=(135,175),blur_radius=40,feather=12,inertia=0.85,anticipate=5)),
 ('20 Direct / wide, shallow mask', dict(trace='direct',axes=(125,105),blur_radius=24,feather=8)),
]
TILE_W, TILE_H, HEADER = 640, 548, 424


def probe(path):
    return json.loads(subprocess.check_output([
        'ffprobe','-v','error','-select_streams','v:0','-show_streams',
        '-of','json',str(path)],text=True))['streams'][0]


def trajectory(path, expected):
    import cv2
    cv2.setNumThreads(1)
    cap = cv2.VideoCapture(str(path))
    try:
        ok, first = cap.read()
        if not ok:
            raise RuntimeError('Cannot read first frame for tracking')
        tracker = cv2.TrackerCSRT_create()
        # Face/head rectangle at half resolution. Displacement is scaled back.
        box = (317,75,78,100)
        tracker.init(cv2.resize(first,(640,360)),box)
        origin = np.array([box[0]+box[2]/2,box[1]+box[3]/2])
        positions = [np.zeros(2)]
        failed = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            success, bounds = tracker.update(cv2.resize(frame,(640,360)))
            if success:
                x,y,w,h = bounds
                delta = 2*(np.array([x+w/2,y+h/2])-origin)
                positions.append(delta)
            else:
                positions.append(positions[-1].copy())
                failed += 1
            if len(positions)%120 == 0:
                print(f'Tracked {len(positions)} frames',flush=True)
    finally:
        cap.release()
    if len(positions) < expected:
        missing = expected - len(positions)
        if missing > 3:
            raise RuntimeError(
                f'Tracking decoded {len(positions)} frames; expected {expected}'
            )
        print(
            f'OpenCV decoded {len(positions)}/{expected} frames; '
            f'padding {missing} missing frame(s) with last tracked position.',
            flush=True
        )
        positions.extend([positions[-1].copy() for _ in range(missing)])

    elif len(positions) > expected:
        raise RuntimeError(
            f'Tracking decoded {len(positions)} frames; expected {expected}'
        )
    values = np.array(positions)
    print(f'Tracking failures: {failed}; displacement range x={values[:,0].min():.1f}..{values[:,0].max():.1f}, y={values[:,1].min():.1f}..{values[:,1].max():.1f} px',flush=True)
    return values


def motion(raw, settings):
    if settings['trace']=='none':
        return np.zeros_like(raw)
    if settings['trace']=='direct':
        return raw.copy()
    result = np.empty_like(raw)
    last = raw[0].copy()
    for i in range(len(raw)):
        target = raw[min(len(raw)-1,i+settings['anticipate'])]
        last = settings['inertia']*last+(1-settings['inertia'])*target
        result[i] = last
    return result


def blur(frame, settings, displacement):
    cx = round(710+settings['xshift']+displacement[0])
    cy = round(255+settings['yshift']+displacement[1])
    ax,ay = settings['axes']
    margin = math.ceil(4*(settings['blur_radius']+settings['feather'])+8)
    box = (max(0,cx-ax-margin),max(0,cy-ay-margin),
           min(frame.width,cx+ax+margin+1),min(frame.height,cy+ay+margin+1))
    result = frame.copy()
    if box[2]<=box[0] or box[3]<=box[1]:
        return result
    patch = frame.crop(box)
    mask = Image.new('L',patch.size,0)
    x,y = cx-box[0],cy-box[1]
    ImageDraw.Draw(mask).ellipse((x-ax,y-ay,x+ax,y+ay),fill=255)
    if settings['feather']:
        mask = mask.filter(ImageFilter.GaussianBlur(settings['feather']))
    blurred = patch.filter(ImageFilter.GaussianBlur(settings['blur_radius']))
    result.paste(Image.composite(blurred,patch,mask),box[:2])
    return result


def command_lines(s, video=SOURCE):
    lines = ['python anonymize_videos.py \\',
             f'  {shlex.quote(str(video))} \\',
             f"  --trace {s['trace']} --xshift {s['xshift']} --yshift {s['yshift']} \\",
             f"  --axes {s['axes'][0]} {s['axes'][1]} --blur_radius {s['blur_radius']} \\"]
    end = f"  --feather {s['feather']}"
    if s['trace']=='smooth':
        end += f" --inertia {s['inertia']} --anticipate {s['anticipate']}"
    lines.append(end)
    return lines


def text(draw,xy,value,size=19,color='white'):
    font = ImageFont.load_default(size=size)
    if draw.textbbox((0,0),value,font=font)[2]>616:
        raise RuntimeError(f'Text exceeds panel: {value}')
    draw.text(xy,value,font=font,fill=color)


def label(title,s,video=SOURCE):
    panel = Image.new('RGB',(640,TILE_H-360),'#142233')
    draw = ImageDraw.Draw(panel)
    text(draw,(12,8),title,21)
    text(draw,(12,37),'Run from folder containing this script and source:',16,'#c7d6e5')
    for i,line in enumerate(command_lines(s,video)):
        text(draw,(12,60+i*24),line,19,'#a6f0ba')
    return panel


def header():
    panel = Image.new('RGB',(2560,HEADER),'#142233')
    draw = ImageDraw.Draw(panel)
    text(draw,(980,10),'ORIGINAL / synchronized / unblurred',21)
    left = ['20 ways to blur the same video','All panels play from the first source frame.',
            'No introductory still or freeze.', '','Commands recreate full-size individual videos.',
            'Required video argument:','Homogeneity1_TargetPremise(Positive).mp4',
            'Default output: anonymized.mp4', 'Change output: --output FILE.mp4']
    right = ['--trace none: fixed ellipse','--trace direct: track every frame',
             '--trace smooth: track with inertia','', '--inertia 0..0.99: higher = slower response',
             '--anticipate N: target N frames ahead', '--axes: ellipse half-width and half-height',
             '--xshift / --yshift: right / down in pixels', '--blur_radius: blur strength; --feather: soft edge']
    for x,lines in [(26,left),(1720,right)]:
        for i,line in enumerate(lines):
            text(draw,(x,45+i*33),line,20,'#c7d6e5')
    return panel


def read_frame(stream,count):
    data = bytearray()
    while len(data)<count:
        part=stream.read(count-len(data))
        if not part:
            break
        data.extend(part)
    if data and len(data)!=count:
        raise RuntimeError('Incomplete decoded frame')
    return bytes(data)


def main():
    parser=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('video',type=Path,help='Path of the video to anonymize')
    parser.add_argument('--output',type=Path)
    parser.add_argument('--grid',action='store_true',help='Render all 20 examples plus moving original')
    parser.add_argument('--trace',choices=('none','direct','smooth'),default='none')
    parser.add_argument('--xshift',type=int,default=0,help='Positive shifts right, in source pixels')
    parser.add_argument('--yshift',type=int,default=0,help='Positive shifts down, in source pixels')
    parser.add_argument('--axes',type=int,nargs=2,default=(105,135),metavar=('X','Y'))
    parser.add_argument('--blur_radius',type=float,default=24)
    parser.add_argument('--feather',type=float,default=4)
    parser.add_argument('--inertia',type=float,default=0.8,help='Previous position weight, 0 <= value < 1')
    parser.add_argument('--anticipate',type=int,default=0,help='Use tracked target N frames ahead; smooth mode only')
    args=parser.parse_args()
    if not 0<=args.inertia<1 or args.anticipate<0 or min(args.axes)<=0 or args.blur_radius<0 or args.feather<0:
        parser.error('Invalid mask, radius, inertia or anticipation parameter')
    if not args.grid and args.trace!='smooth' and args.anticipate:
        parser.error('--anticipate requires --trace smooth')
    output=args.output or Path('anonymization_grid.mp4' if args.grid else 'anonymized.mp4')
    if output.exists():
        parser.error(f'Output exists: {output}; choose another --output')
    if output.suffix.lower()!='.mp4' or not args.video.is_file():
        parser.error('Input must exist and output must end in .mp4')
    if not all(shutil.which(t) for t in ('ffmpeg','ffprobe')):
        parser.error('FFmpeg and ffprobe are required on PATH')
    meta=probe(args.video)
    size=(int(meta['width']),int(meta['height']))
    if size!=(1280,720):
        parser.error('These masks are calibrated for 1280x720')
    fps=meta['avg_frame_rate']
    if meta['r_frame_rate']!=fps:
        parser.error('Constant frame rate required')
    expected=int(meta['nb_frames'])
    examples=[(title,dict(DEFAULTS,**s)) for title,s in EXAMPLES] if args.grid else [('Individual', {k:getattr(args,k) for k in DEFAULTS})]
    raw=trajectory(args.video,expected) if any(s['trace']!='none' for _,s in examples) else np.zeros((expected,2))
    paths=[motion(raw,s) for _,s in examples]
    labels=[label(title,s,args.video) for title,s in examples] if args.grid else []
    backdrop=header() if args.grid else None
    output_size=(2560,HEADER+5*TILE_H) if args.grid else size
    output.parent.mkdir(parents=True,exist_ok=True)
    writer = subprocess.Popen(['ffmpeg', '-v', 'error', '-n', '-f', 'rawvideo', '-pix_fmt', 'rgb24',
                               '-s', f'{output_size[0]}x{output_size[1]}', '-r', fps, '-i', '-',
                               '-i', str(args.video),
                               '-map', '0:v:0', '-map', '1:a?',
                               '-c:v', 'libx264', '-preset', 'fast', '-crf', '18', '-threads', '2', '-pix_fmt',
                               'yuv420p',
                               '-c:a', 'copy',
                               '-map_metadata', '-1', '-movflags', '+faststart', str(output)], stdin=subprocess.PIPE)
    decoder=subprocess.Popen(['ffmpeg','-v','error','-i',str(args.video),'-map','0:v:0','-vsync','0',
        '-f','rawvideo','-pix_fmt','rgb24','-threads','1','-'],stdout=subprocess.PIPE)
    count=0
    try:
        while True:
            data=read_frame(decoder.stdout,size[0]*size[1]*3)
            if not data:
                break
            if count>=expected:
                raise RuntimeError('More frames than expected')
            original=Image.frombytes('RGB',size,data)
            if args.grid:
                result=Image.new('RGB',output_size,'#142233')
                result.paste(backdrop,(0,0))
                result.paste(original.resize((640,360),Image.Resampling.LANCZOS),(960,45))
                for i,(_,s) in enumerate(examples):
                    frame=blur(original,s,paths[i][count])
                    x,y=(i%4)*640,HEADER+(i//4)*TILE_H
                    result.paste(frame.resize((640,360),Image.Resampling.LANCZOS),(x,y))
                    result.paste(labels[i],(x,y+360))
            else:
                result=blur(original,examples[0][1],paths[0][count])
            writer.stdin.write(result.tobytes())
            count+=1
            if count%60==0:
                print(f'Rendered {count}/{expected} frames',flush=True)
        if decoder.wait()!=0:
            raise RuntimeError('Decoding failed')
    finally:
        decoder.stdout.close()
        if decoder.poll() is None:
            decoder.terminate()
            decoder.wait()
        writer.stdin.close()
        code=writer.wait()
    if code:
        raise RuntimeError('Encoding failed')
    check = probe(output)
    output_frames = int(check['nb_frames'])

    if output_frames != count or Fraction(check['avg_frame_rate']) != Fraction(fps):
        raise RuntimeError(
            f'Output verification failed: rendered={count}, '
            f'output={output_frames}, metadata_expected={expected}'
        )

    if count != expected:
        print(
            f'Warning: source metadata reports {expected} frames, '
            f'but FFmpeg decoded {count}. Output contains all {count} decoded frames.',
            flush=True
        )
    print(f'Saved {output}: {count} frames, {fps} fps, silent; no added introduction.',flush=True)


if __name__=='__main__':
    main()
