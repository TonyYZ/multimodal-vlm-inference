"""Audio-video model adapters shared by the animation and gesture experiments."""

import torch

MINICPMO_REVISION = "eb4141de24550a3a1a4082f718be58f46b761728"


def load_qwen_omni(model_id: str):
    import transformers

    prefix = "Qwen3OmniMoe" if "qwen3-omni" in model_id.lower() else "Qwen2_5Omni"
    model = getattr(transformers, prefix + "ForConditionalGeneration").from_pretrained(
        model_id, dtype="auto", device_map="auto", attn_implementation="sdpa",
    )
    model.disable_talker()
    processor = getattr(transformers, prefix + "Processor").from_pretrained(model_id)
    return model, processor


def run_qwen_omni_animation(model, processor, content, max_new_tokens, print_video_shapes=False):
    from qwen_omni_utils import process_mm_info

    messages = [{"role": "user", "content": content}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    audio, images, videos, video_kwargs = process_mm_info(
        messages, use_audio_in_video=False, return_video_kwargs=True,
        image_patch_size=processor.video_processor.patch_size,
    )
    if audio is not None and len(audio):
        raise RuntimeError("Animation experiments must not supply audio.")
    sampled_fps = video_kwargs.get("fps", [])
    kwargs = {"fps": sampled_fps[0], "do_sample_frames": False} if videos else {}
    inputs = processor(
        text=text, images=images, videos=videos, return_tensors="pt", padding=True,
        use_audio_in_video=False, videos_kwargs=kwargs,
    )
    if videos:
        # With audio disabled the token expansion is fps-independent. Preserve each
        # clip's actual timing rather than imposing one fps on all split clips.
        inputs["video_second_per_grid"] = torch.tensor([
            processor.video_processor.temporal_patch_size / fps for fps in sampled_fps
        ])
    if print_video_shapes:
        print("video input shapes:", [tuple(video.shape) for video in videos or []])
        print("video sampled fps:", sampled_fps)
    inputs = inputs.to(model.device).to(model.dtype)
    with torch.inference_mode():
        output = model.generate(
            **inputs, thinker_max_new_tokens=max_new_tokens,
            thinker_do_sample=False, return_audio=False, use_audio_in_video=False,
        )
    if isinstance(output, tuple):
        output = output[0]
    if hasattr(output, "sequences"):
        output = output.sequences
    length = inputs.input_ids.shape[1]
    if not torch.equal(output[:, :length], inputs.input_ids):
        raise RuntimeError("Cannot safely trim an output without its input prefix.")
    return processor.batch_decode(output[:, length:], skip_special_tokens=True)[0].strip()


def load_minicpmo(model_id: str):
    import transformers

    if transformers.__version__ != "4.51.0":
        raise RuntimeError("MiniCPM-o requires its separate environment (requirements_minicpmo.txt).")
    model = transformers.AutoModel.from_pretrained(
        model_id, revision=MINICPMO_REVISION, trust_remote_code=True,
        torch_dtype=torch.bfloat16, attn_implementation="sdpa",
        init_vision=True, init_audio=True, init_tts=False,
    ).eval().cuda()
    model.prepare_processor()
    return model, model.processor


def run_minicpmo(model, content, max_new_tokens, *, system_message=None, use_audio=False, audit=None):
    import numpy as np
    from PIL import Image

    native = []
    frame_count = 0
    audio_seconds = 0.0
    for item in content:
        if item["type"] == "text":
            native.append(item["text"])
        elif item["type"] == "video" and use_audio:
            from minicpmo.utils import get_video_frame_audio_segments

            fps = float(item.get("fps", 3))
            if not fps.is_integer():
                raise ValueError("MiniCPM-o's native stacked-frame audio-video input requires integer fps.")
            frames, audio, stacked = get_video_frame_audio_segments(
                item["video"], stack_frames=int(fps), use_ffmpeg=True, use_audio=True,
            )
            if audio is None or len(audio) != len(frames):
                raise RuntimeError("MiniCPM-o did not decode aligned video/audio segments.")
            if any(not len(segment) or not np.isfinite(segment).all() for segment in audio):
                raise RuntimeError("MiniCPM-o decoded empty/nonfinite audio segments.")
            for i, frame in enumerate(frames):
                native.extend([frame, audio[i]])
                frame_count += 1
                audio_seconds += len(audio[i]) / 16000
                if stacked and stacked[i] is not None:
                    native.append(stacked[i])
                    frame_count += int(fps) - 1
        elif item["type"] == "video":
            from qwen_vl_utils import process_vision_info

            _, videos = process_vision_info([{"role": "user", "content": [item]}])
            for frame in videos[0].to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy():
                native.append(Image.fromarray(frame))
                frame_count += 1
        else:
            raise ValueError(f"Unsupported MiniCPM-o content: {item['type']}")
    # Cap image area without cropping; the model retains its native token compression.
    max_pixels = min((item.get("max_pixels", 307200) for item in content if item["type"] == "video"), default=307200)
    for image in native:
        if isinstance(image, Image.Image) and image.width * image.height > max_pixels:
            factor = (max_pixels / (image.width * image.height)) ** 0.5
            image.thumbnail((max(1, int(image.width * factor)), max(1, int(image.height * factor))))
    if use_audio and (frame_count == 0 or audio_seconds <= 0):
        raise RuntimeError("MiniCPM-o requires both video frames and audio for the gesture study.")
    messages = [] if system_message is None else [{"role": "system", "content": [system_message]}]
    messages.append({"role": "user", "content": native})
    with torch.inference_mode():
        response = model.chat(
            msgs=messages, max_new_tokens=max_new_tokens, max_inp_length=32768,
            do_sample=False, enable_thinking=False, generate_audio=False,
            omni_mode=use_audio, use_tts_template=use_audio,
            max_slice_nums=1, use_image_id=False,
        )
    if not isinstance(response, str):
        raise RuntimeError(f"Unexpected MiniCPM-o answer type: {type(response)}")
    if audit is not None:
        audit.update(
            audit_level="native API inputs, not internal embedding tensors",
            video_frames=frame_count, audio_duration_s=audio_seconds,
            modality_checks_passed=frame_count > 0 and (not use_audio or audio_seconds > 0),
            native_answer_only=True,
            token_budget_reached=len(model.processor.tokenizer.encode(response, add_special_tokens=False)) >= max_new_tokens,
            budget_check="decoded text only", model_revision=MINICPMO_REVISION,
        )
    return response.strip()
