from __future__ import annotations

import argparse
import csv
import json
import os
import re
from pathlib import Path
from statistics import mean


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_GOLD = PROJECT_ROOT / "processed" / "gold_descriptions.json"
DEFAULT_DESCRIPTION_DIR = PROJECT_ROOT / "results" / "descriptions"
DEFAULT_OUTPUT = PROJECT_ROOT / "results" / "description_judgments.jsonl"
DEFAULT_SUMMARY = PROJECT_ROOT / "results" / "description_judgment_summary.csv"
JUDGE_PROMPT_VERSION = "4"
COMPONENT_FIELDS = (
    "text_score",
    "object_score",
    "spatial_score",
    "temporal_score",
    "event_type_score",
)
HALLUCINATION_PENALTY_POINTS = 5
OVERALL_SCORE_METHOD = (
    "10 * sum(component_scores) - 5 * hallucination_penalty, clamped to 0-100"
)

SYSTEM_PROMPT = """You are judging whether a VLM accurately described experimental materials.
Judge perceptual accuracy only. Ignore whether the model's pragmatic inference is correct.
Do not penalize differences in event segmentation alone.
The model may merge multiple gold events or split one gold event into several, provided that all objects,
locations, event types, and chronological relationships are accurately preserved. If merging or splitting
causes information loss or incorrect ordering, penalize those specific errors instead.
Evaluate the global timeline, including whether each visual event occurs before or after the
correct text segments. Penalize incorrect relative text-visual order.
Return valid JSON only."""

USER_PROMPT = """Gold description:
{gold}

Model description:
{model_response}

Score the model description against the gold description.
Use this JSON schema:
{{
  "text_score": 0,
  "object_score": 0,
  "spatial_score": 0,
  "temporal_score": 0,
  "event_type_score": 0,
  "hallucination_penalty": 0,
  "major_errors": ["short error descriptions"]
}}

Scoring anchors:
- 2 = mostly correct, 1 = partly correct, 0 = mostly wrong or missing.
- hallucination_penalty: 0 = none, 1 = minor, 2 = major.
- Do not return an overall score; it is calculated deterministically from these ratings."""


def load_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def source_key(path: str) -> str:
    return Path(path.replace("\\", "/")).stem


def extract_json(text: str) -> dict | None:
    text = text.strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, flags=re.DOTALL)
    if fenced:
        text = fenced.group(1).strip()
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            return None
        try:
            parsed = json.loads(match.group(0))
        except json.JSONDecodeError:
            return None
    return parsed if isinstance(parsed, dict) else None


def build_gold_lookup(path: Path) -> dict[str, dict]:
    return {source_key(row["source_video"]): row for row in load_json(path)}


def deterministic_overall_score(judgment: dict) -> int:
    values = [judgment.get(field) for field in COMPONENT_FIELDS]
    penalty = judgment.get("hallucination_penalty")
    for field, value in zip((*COMPONENT_FIELDS, "hallucination_penalty"), (*values, penalty)):
        if isinstance(value, bool) or value not in (0, 1, 2):
            raise ValueError(f"{field} must be 0, 1, or 2; got {value!r}")

    score = 10 * sum(values) - HALLUCINATION_PENALTY_POINTS * penalty
    return max(0, min(100, score))


def perceptual_gold(gold: dict) -> dict:
    evidence = {
        "source_video": gold["source_video"],
    }
    if gold.get("timeline"):
        evidence["timeline"] = gold["timeline"]
    else:
        evidence["text_segments"] = gold["text_segments"]
        evidence["animation_descriptions"] = gold["animation_descriptions"]
    return evidence


def description_files(paths: list[Path] | None) -> list[Path]:
    if paths:
        return paths
    return sorted(DEFAULT_DESCRIPTION_DIR.glob("*_descriptions.jsonl"))


def judge_with_openai(model: str, gold: dict, model_response: str) -> str:
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise ImportError("Install the OpenAI SDK first: pip install openai") from exc

    client = OpenAI()
    response = client.responses.create(
        model=model,
        temperature=0,
        input=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": USER_PROMPT.format(
                    gold=json.dumps(gold, ensure_ascii=False, indent=2),
                    model_response=model_response,
                ),
            },
        ],
    )
    return response.output_text.strip()


def write_summary(rows: list[dict], path: Path) -> None:
    grouped: dict[tuple[str, str], list[float]] = {}
    for row in rows:
        parsed = row.get("judgment_json") or {}
        score = parsed.get("overall_score")
        if isinstance(score, (int, float)):
            key = (row["model"], row["input_mode"])
            grouped.setdefault(key, []).append(float(score))

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["model", "input_mode", "n", "mean_overall_score"],
        )
        writer.writeheader()
        for (model, input_mode), scores in sorted(grouped.items()):
            writer.writerow(
                {
                    "model": model,
                    "input_mode": input_mode,
                    "n": len(scores),
                    "mean_overall_score": mean(scores),
                }
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Judge VLM descriptions against gold material descriptions."
    )
    parser.add_argument("--descriptions", type=Path, nargs="+", default=None)
    parser.add_argument("--gold", type=Path, default=DEFAULT_GOLD)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument(
        "--judge-model",
        default=os.environ.get("OPENAI_JUDGE_MODEL", "gpt-4.1-mini"),
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--input-modes",
        choices=("split", "pure-video", "pure-text"),
        nargs="+",
        default=None,
        help="Only judge descriptions from these input modes.",
    )
    parser.add_argument(
        "--description-prompt-version",
        default="3",
        help="Only judge descriptions from this prompt version; use 'all' to disable.",
    )
    parser.add_argument(
        "--dry-run-prompts",
        action="store_true",
        help="Write judge prompts without calling the judge model.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.dry_run_prompts and not (
        os.environ.get("OPENAI_API_KEY") or os.environ.get("OPENAI_ADMIN_KEY")
    ):
        raise EnvironmentError(
            "Set OPENAI_API_KEY in this terminal before running the judge."
        )
    gold_by_source = build_gold_lookup(args.gold)
    files = description_files(args.descriptions)
    if not files:
        raise FileNotFoundError(f"No description files found in {DEFAULT_DESCRIPTION_DIR}")

    jobs: list[tuple[Path, dict, dict]] = []
    for path in files:
        for row in load_jsonl(path):
            if args.input_modes and row.get("input_mode") not in args.input_modes:
                continue
            if (
                args.description_prompt_version != "all"
                and str(row.get("prompt_version"))
                != args.description_prompt_version
            ):
                continue
            gold = gold_by_source.get(source_key(row["source_video"]))
            if gold and "response" in row:
                jobs.append((path, row, gold))
    if args.limit is not None:
        jobs = jobs[: args.limit]
    if not jobs:
        raise ValueError("No judgeable description rows found.")

    print(f"Found {len(jobs)} description row(s) to judge.")
    judged_rows: list[dict] = []
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        for path, row, gold in jobs:
            print(f"Judging: {row['source_video']} [{row['model']} / {row['input_mode']}]")
            gold_evidence = perceptual_gold(gold)
            error = None
            try:
                if args.dry_run_prompts:
                    judgment_raw = USER_PROMPT.format(
                        gold=json.dumps(gold_evidence, ensure_ascii=False, indent=2),
                        model_response=row["response"],
                    )
                else:
                    judgment_raw = judge_with_openai(
                        args.judge_model,
                        gold=gold_evidence,
                        model_response=row["response"],
                    )
            except Exception as exc:
                judgment_raw = ""
                error = repr(exc)

            judgment_json = None if args.dry_run_prompts else extract_json(judgment_raw)
            if judgment_json is not None:
                try:
                    judgment_json["overall_score"] = deterministic_overall_score(
                        judgment_json
                    )
                except ValueError as exc:
                    error = str(exc)

            judged = {
                "description_file": str(path),
                "judge_model": args.judge_model,
                "judge_prompt_version": JUDGE_PROMPT_VERSION,
                "model": row["model"],
                "input_mode": row["input_mode"],
                "description_prompt_version": row.get("prompt_version"),
                "source_video": row["source_video"],
                "gold": gold_evidence,
                "model_response": row["response"],
                "judgment_raw": judgment_raw,
                "judgment_json": judgment_json,
                "overall_score_method": OVERALL_SCORE_METHOD,
            }
            if error:
                judged["error"] = error
                print(f"Error: {error}")
            judged_rows.append(judged)
            f.write(json.dumps(judged, ensure_ascii=False) + "\n")
            f.flush()

    if not args.dry_run_prompts:
        write_summary(judged_rows, args.summary)
        print(f"Wrote summary to {args.summary}")
        successful = sum(
            row.get("judgment_json") is not None and "error" not in row
            for row in judged_rows
        )
        print(f"Successful judgments: {successful}/{len(judged_rows)}")
        if successful == 0:
            raise RuntimeError(
                "All judge calls failed. Inspect the error field in the judgment JSONL."
            )
    print(f"Wrote judgments to {args.output}")


if __name__ == "__main__":
    main()
