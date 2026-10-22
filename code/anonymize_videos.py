from pathlib import Path
import cv2

def cartoonize(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 11)
    edges = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY,
        15, 8
    )
    color = cv2.bilateralFilter(frame, 9, 300, 300)
    return cv2.bitwise_and(color, color, mask=edges)


# anonymize_videos.py is inside ...\multimodal-vlm-inference\code
project_dir = Path(__file__).resolve().parent.parent

input_path = project_dir / "materials" / "gestures" / "Homogeneity1_TargetPremise(Positive).mp4"
output_path = project_dir / "materials" / "gestures" / "cartoon_noaudio.mp4"

print("Input:", input_path)
print("Exists:", input_path.exists())

cap = cv2.VideoCapture(str(input_path))

if not cap.isOpened():
    raise RuntimeError(f"Could not open video: {input_path}")

fps = cap.get(cv2.CAP_PROP_FPS)
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

out = cv2.VideoWriter(
    str(output_path),
    cv2.VideoWriter_fourcc(*"mp4v"),
    fps,
    (w, h)
)

while True:
    ok, frame = cap.read()
    if not ok:
        break

    out.write(cartoonize(frame))

cap.release()
out.release()

print("Saved:", output_path)