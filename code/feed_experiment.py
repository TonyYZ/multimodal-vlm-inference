from pathlib import Path
import json
import torch

from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info


MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"

PROJECT_ROOT = Path(__file__).resolve().parents[1]
VIDEO_DIR = PROJECT_ROOT / "materials" / "animations"
RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

OUTPUT_FILE = RESULTS_DIR / "qwen2_5_vl_3b_outputs.jsonl"


def load_model():
    """
    The first time this runs, Hugging Face will automatically download the model
    into the local HF cache.
    """
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_ID,
        torch_dtype="auto",
        device_map="auto",
    )

    processor = AutoProcessor.from_pretrained(MODEL_ID)

    return model, processor


def run_one_video(model, processor, video_path: Path, prompt: str) -> str:
    """
    Run Qwen2.5-VL on one local video.
    video_path.as_uri() converts:
        /Users/tony/.../video.mp4
    into:
        file:///Users/tony/.../video.mp4
    which Qwen expects for local video paths.
    """

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": str(video_path.resolve()),
                    "fps": 1.0,
                    "max_pixels": 360 * 420,
                },
                {
                    "type": "text",
                    "text": prompt,
                },
            ],
        }
    ]

    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    image_inputs, video_inputs = process_vision_info(messages)

    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )

    inputs = inputs.to(model.device)

    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=128,
            do_sample=False,
        )

    generated_ids_trimmed = [
        output_ids[len(input_ids):]
        for input_ids, output_ids in zip(inputs.input_ids, generated_ids)
    ]

    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]

    return output_text


def main():
    if not VIDEO_DIR.exists():
        raise FileNotFoundError(f"Cannot find video folder: {VIDEO_DIR}")

    video_files = sorted(
        list(VIDEO_DIR.glob("*.mp4"))
        + list(VIDEO_DIR.glob("*.m4v"))
        + list(VIDEO_DIR.glob("*.mov"))
        + list(VIDEO_DIR.glob("*.webm"))
        + list(VIDEO_DIR.glob("*.avi"))
    )

    if not video_files:
        raise FileNotFoundError(f"No video files found in {VIDEO_DIR}")

    print(f"Found {len(video_files)} videos.")
    print(f"Loading model: {MODEL_ID}")

    model, processor = load_model()
    print("CUDA available:", torch.cuda.is_available())
    print("Model device:", next(model.parameters()).device)

    prompt = (
        "Watch the video carefully. Transcribe all visible text in the video "
        "in the order it appears."
    )

    with OUTPUT_FILE.open("w", encoding="utf-8") as f:
        for video_path in video_files:
            print(f"Running: {video_path.name}")

            try:
                response = run_one_video(model, processor, video_path, prompt)

                record = {
                    "model": MODEL_ID,
                    "video_file": str(video_path.relative_to(PROJECT_ROOT)),
                    "prompt": prompt,
                    "response": response,
                }

                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                print(response)
                print("-" * 80)

            except Exception as e:
                record = {
                    "model": MODEL_ID,
                    "video_file": str(video_path.relative_to(PROJECT_ROOT)),
                    "prompt": prompt,
                    "error": repr(e),
                }

                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                print(f"Error on {video_path.name}: {e}")


if __name__ == "__main__":
    main()