from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STORY_FILE = PROJECT_ROOT / "materials" / "stories" / "TSC_EXP_2_Stories.rtf"
DEFAULT_OUTPUT = PROJECT_ROOT / "processed" / "gold_descriptions.json"

TEXT_REPLACEMENTS = {
    "\u2018": "'",
    "\u2019": "'",
    "\u201c": '"',
    "\u201d": '"',
    "\u2026": "...",
    "\ufffd": "",
}

TASK_HEADING = re.compile(
    r"^(?:Homogeneity|Presupposition|ScalarImplicature|Supplement)\d+_"
    r"(?:TargetPremise|ControlPremise)(?:\([^)]+\))?$"
)
INFERENCE_HEADING = re.compile(r"^(Target|Control) inference\s*:\s*(.*)$")
ANIMATION_MARKER = re.compile(r"\[(?:[^\]]*animation[^\]]*)\]\s*", re.IGNORECASE)
ITALIC_LINE = re.compile(r"^\s*\*(.+)\*\s*$")


def rtf_to_text(path: Path) -> str:
    raw = path.read_text(encoding="cp1252")

    def decode_hex(match: re.Match[str]) -> str:
        return bytes.fromhex(match.group(1)).decode("cp1252")

    text = re.sub(r"\\'([0-9a-fA-F]{2})", decode_hex, raw)
    text = text.replace("\\\n", "\n")
    text = re.sub(r"\\[a-zA-Z]+-?\d* ?", "", text)
    text = text.replace(r"\{", "{").replace(r"\}", "}").replace(r"\\", "\\")
    return text.replace("{", "").replace("}", "").replace("\\", "")


def clean_line(line: str) -> str:
    line = line.strip()
    for old, new in TEXT_REPLACEMENTS.items():
        line = line.replace(old, new)
    line = re.sub(r"^Slide\s+\d+\s*:\s*", "", line)
    line = line.replace("*", "")
    return re.sub(r"\s+", " ", line).strip()


def split_animation_line(line: str) -> tuple[str, str | None]:
    match = ANIMATION_MARKER.search(line)
    if match:
        prefix = clean_line(line[: match.start()])
        animation = clean_line(line[match.end() :])
        return prefix, animation

    match = ITALIC_LINE.match(line)
    if match:
        return "", clean_line(match.group(1))
    return clean_line(line), None


def empty_entry(task_name: str) -> dict:
    return {
        "task_name": task_name,
        "source_video": f"materials\\animations\\{task_name}.m4v",
        "text_segments": [],
        "animation_descriptions": [],
        "timeline": [],
        "target_inference": "",
        "control_inference": "",
    }


def extract_gold(path: Path) -> list[dict]:
    entries: list[dict] = []
    current: dict | None = None
    inference_type: str | None = None

    for raw_line in rtf_to_text(path).splitlines():
        line = clean_line(raw_line)
        if not line:
            continue

        if TASK_HEADING.match(line):
            if current:
                entries.append(current)
            current = empty_entry(line)
            inference_type = None
            continue
        if current is None:
            continue

        inference_match = INFERENCE_HEADING.match(line)
        if inference_match:
            inference_type = inference_match.group(1).lower()
            inline_text = clean_line(inference_match.group(2))
            if inline_text:
                current[f"{inference_type}_inference"] = inline_text
            continue

        if inference_type:
            current[f"{inference_type}_inference"] = line
            inference_type = None
            continue

        text_prefix, animation = split_animation_line(raw_line)
        if text_prefix:
            current["text_segments"].append(text_prefix)
            current["timeline"].append(
                {
                    "order": len(current["timeline"]) + 1,
                    "type": "text",
                    "text": text_prefix,
                }
            )
        if animation:
            current["animation_descriptions"].append(animation)
            current["timeline"].append(
                {
                    "order": len(current["timeline"]) + 1,
                    "type": "visual",
                    "description": animation,
                }
            )

    if current:
        entries.append(current)
    return entries


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract text and animation gold descriptions from TSC_EXP_2_Stories.rtf."
    )
    parser.add_argument("--story-file", type=Path, default=DEFAULT_STORY_FILE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    entries = extract_gold(args.story_file)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(entries, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {len(entries)} gold descriptions to {args.output}")


if __name__ == "__main__":
    main()
