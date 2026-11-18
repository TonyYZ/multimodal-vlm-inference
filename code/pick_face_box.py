#!/usr/bin/env python3
"""Interactively pick a face bounding box from the first frame of a video.

Usage (Windows CMD):
  python pick_face_box.py "video.mov"

Requirements: tkinter (normally shipped with Windows Python), Pillow, OpenCV.
Drag a rectangle around the face, press Enter to copy the --face-box argument.
Coordinates are returned in the ORIGINAL video resolution even if the window is scaled.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import sys
import tkinter as tk
from PIL import Image, ImageTk
import cv2


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('video', type=Path)
    args = p.parse_args()
    if not args.video.is_file():
        p.error(f'Video does not exist: {args.video}')

    cap = cv2.VideoCapture(str(args.video))
    try:
        ok, frame = cap.read()
    finally:
        cap.release()
    if not ok:
        p.error('Unable to read the first frame of this video')

    height, width = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    original = Image.fromarray(rgb)
    root = tk.Tk()
    root.title(f'Select face box — {args.video.name} ({width} × {height})')

    # Fit within most laptop screens while keeping full resolution coordinates.
    max_w = max(400, min(1150, root.winfo_screenwidth() - 100))
    max_h = max(300, min(750, root.winfo_screenheight() - 170))
    scale = min(1.0, max_w / width, max_h / height)
    display_w, display_h = max(1, round(width * scale)), max(1, round(height * scale))
    preview = original.resize((display_w, display_h), Image.Resampling.LANCZOS)
    photo = ImageTk.PhotoImage(preview, master=root)

    hint = tk.Label(root, text='Drag a rectangle around the FACE (forehead to chin). Press Enter to copy, Esc to exit.',
                    padx=12, pady=9)
    hint.pack()
    canvas = tk.Canvas(root, width=display_w, height=display_h, cursor='crosshair', highlightthickness=0)
    canvas.pack()
    canvas.create_image(0, 0, image=photo, anchor='nw')
    result_text = tk.StringVar(value='Click and drag to select the face.')
    result_label = tk.Label(root, textvariable=result_text, font=('Consolas', 12), padx=12, pady=9)
    result_label.pack()

    state = {'start': None, 'rectangle': None, 'box': None}

    def clamp(x, upper):
        return max(0, min(int(x), upper))

    def begin(event):
        x, y = clamp(event.x, display_w), clamp(event.y, display_h)
        state['start'] = (x, y)
        state['box'] = None
        if state['rectangle'] is not None:
            canvas.delete(state['rectangle'])
        state['rectangle'] = canvas.create_rectangle(x, y, x, y, outline='#ff3333', width=3)

    def move(event):
        if state['start'] is None:
            return
        x, y = clamp(event.x, display_w), clamp(event.y, display_h)
        x0, y0 = state['start']
        canvas.coords(state['rectangle'], x0, y0, x, y)

    def end(event):
        if state['start'] is None:
            return
        x0, y0 = state['start']
        x1, y1 = clamp(event.x, display_w), clamp(event.y, display_h)
        state['start'] = None
        left, right = sorted((x0, x1))
        top, bottom = sorted((y0, y1))
        # Convert boundaries rather than independent widths, preventing scale rounding drift.
        src_left = round(left * width / display_w)
        src_right = round(right * width / display_w)
        src_top = round(top * height / display_h)
        src_bottom = round(bottom * height / display_h)
        w, h = src_right-src_left, src_bottom-src_top
        if w < 8 or h < 8:
            state['box'] = None
            result_text.set('Selection too small. Drag a larger box around the face.')
        else:
            state['box'] = (src_left, src_top, w, h)
            result_text.set(f'--face-box {src_left} {src_top} {w} {h}   (press Enter to copy)')

    def confirm(event=None):
        if state['box'] is None:
            result_text.set('First drag a box around the face.')
            return
        arg = '--face-box ' + ' '.join(map(str, state['box']))
        root.clipboard_clear()
        root.clipboard_append(arg)
        root.update()
        print('\n' + arg, flush=True)
        result_text.set(arg + '   — copied to clipboard!')

    canvas.bind('<ButtonPress-1>', begin)
    canvas.bind('<B1-Motion>', move)
    canvas.bind('<ButtonRelease-1>', end)
    root.bind('<Return>', confirm)
    root.bind('<Escape>', lambda event: root.destroy())
    tk.Button(root, text='Copy face-box', command=confirm, padx=16, pady=5).pack(pady=(0, 9))
    root.mainloop()

if __name__ == '__main__':
    main()
