# Gesture Perception Pilot Review

## Scope

One video: `Homogeneity1_TargetPremise(Negative).mp4` from the anonymized folder.
This is a diagnostic inspection, not an accuracy estimate for all materials.
No judge model was called. The VLMs received no written story or target/baseline
question. Existing inference ratings and structured descriptions are unchanged.

## Input And Output Checks

- All four processors produce nonempty, finite audio and visual features.
- The source video stream is 14.2 seconds; the decoded audio is approximately
  14.25 seconds at 16 kHz. The slight difference from the container's 14.288-second
  duration is within codec-padding tolerance.
- Qwen receives 42 sampled frames covering the clip. The corrected Gemma pilot
  also receives 42 frames. Original Gemma runs used 32 uniformly sampled frames
  because the processor's frame-count default overrode fps.
- Generated prompt prefixes match input token IDs, supporting the answer-trimming
  implementation. The three Qwen plain-text pilots finish below the output cap.
  Gemma reaches the 768-token cap while repeating gesture descriptions.

## Speech Cross-Check

An independent `openai/whisper-small.en` run on the original soundtrack produces:

> Sam is participating in a treasure hunt in the forest and she's looking for
> crosses and coins. Very quickly, Sam will find and Sam will not

Ignoring punctuation and capitalization, all three Qwen plain-text pilot speech
outputs have the same word sequence as this independent ASR result. The apparent
sentence fragments occur around silent gestures; no gesture glosses should be
inserted as if they were spoken. The documented story also supports this wording.
This is corroboration, not a human-verified gold transcript. Gemma's opening
differs substantially from both the documented story and the independent ASR.

## Direct Visual Inspection

Inspection of source frames shows perpendicular index-finger configurations
forming cross-like shapes, rounded/pinched-finger configurations repeated at
different locations, and a final two-handed lateral reach followed by closing
and drawing the hands inward. These observations concern physical movements,
not inferred meanings such as taking all objects. Exact repetitions and timing
still need a checked event reference.

| Model | Combined speech-and-gesture prompt |
| --- | --- |
| Qwen2.5-Omni 3B | Transcribes speech but omits gestures. |
| Qwen2.5-Omni 7B | Transcribes speech but omits gestures. |
| Qwen3-Omni 30B | Includes gestures, but misses distinguishing cross/rounded-finger configurations and inaccurately describes the movement sequence. |
| Gemma 4 12B | Inaccurate opening transcript; repeats alternating pointing descriptions and reaches the token cap. |

These are qualitative observations from one clip, not numerical model rankings.

## Controlled Visual Follow-Up

`visual_ablation.jsonl` uses the same gesture-only question and sampled video,
with audio present versus absent, for Qwen2.5-Omni 3B and 7B. The 3B model
produces gesture descriptions in both cases, though generic pointing/circular
movement claims miss important observed configurations. The 7B model likewise
describes gestures both with and without audio, with inaccurate/repeated
pointing, fist and circular-movement claims. All four follow-up responses completed.

Thus, omission of gestures under the original two-part prompt is not by itself
evidence that the video input was missing. Both the requested output task and
perceptual accuracy matter. This one-clip check does not establish a general
audio-interference mechanism.

## Before Automated Judging

Verify a gold transcript and physical event sequence, including overlaps and
viewer-relative directions, against the recording. Keep format compliance,
transcription accuracy and gesture accuracy separate. Incorrect or missing model
descriptions must be counted as errors, not removed to improve apparent accuracy.
Do not promote an unchecked RTF or an ASR result to gold automatically.

Evidence: the four `*_sanity.jsonl` files, `independent_asr_reference.json`,
`visual_ablation.jsonl`, and the original anonymized video. Temporary frame
inspection previews are under `tmp/gesture_sanity/`.
