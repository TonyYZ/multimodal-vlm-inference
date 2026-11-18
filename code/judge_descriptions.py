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
DEFAULT_DESCRIPTION_DIR = PROJECT_ROOT / "results" / "animation" / "perception" / "descriptions"
DEFAULT_OUTPUT = PROJECT_ROOT / "results" / "animation" / "perception" / "judgments" / "description_judgments.jsonl"
DEFAULT_SUMMARY = PROJECT_ROOT / "results" / "animation" / "perception" / "judgments" / "description_judgment_summary.csv"
JUDGE_PROMPT_VERSION = "5"

JUDGE_PROMPT = """Here is the correct description of a story shown as several slides, taken from the experiment materials:

{gold}

Here is a model's description of the same slides:

{model_response}

How well does the model's description match the correct description?
Answer with a single integer from 0 to 100.
0 = does not match at all.
100 = matches perfectly.
Output only the number."""


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


def parse_score(text: str) -> int | None:
    match = re.match(r"\s*(\d{1,3})(?:\s|$|\.)", text)
    return int(match.group(1)) if match and int(match.group(1)) <= 100 else None


def perceptual_gold(gold: dict) -> str:
    """The story text and 'Animation:' lines from TSC_EXP_2_Stories.rtf, in their original order."""
    return "\n".join(
        f"Slide {number}: "
        + (entry["text"] if entry["type"] == "text" else f"Animation: {entry['description']}")
        for number, entry in enumerate(gold["timeline"], start=1)
    )


def description_files(paths: list[Path] | None) -> list[Path]:
    if paths:
        return paths
    return sorted(DEFAULT_DESCRIPTION_DIR.glob("*_descriptions.jsonl"))


def judge_with_openai(model: str, gold: str, model_response: str) -> str:
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise ImportError("Install the OpenAI SDK first: pip install openai") from exc

    client = OpenAI()
    response = client.responses.create(
        model=model,
        temperature=0,
        input=[{"role": "user", "content": JUDGE_PROMPT.format(gold=gold, model_response=model_response)}],
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
    parser.add_argument("--resume", action="store_true",
                        help="Reuse successful judgments only when the model response, gold, and judge versions match.")
    parser.add_argument(
        "--input-modes",
        choices=("split", "pure-video", "pure-text"),
        nargs="+",
        default=None,
        help="Only judge descriptions from these input modes.",
    )
    parser.add_argument(
        "--description-prompt-version",
        default="5",
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
    previous = {}
    if args.resume and args.output.exists():
        previous = {(r["model"], r["input_mode"], source_key(r["source_video"])): r
                    for r in load_jsonl(args.output)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("a" if args.resume else "w", encoding="utf-8") as f:
        for path, row, gold in jobs:
            print(f"Judging: {row['source_video']} [{row['model']} / {row['input_mode']}]")
            gold_evidence = perceptual_gold(gold)
            saved = previous.get((row["model"], row["input_mode"], source_key(row["source_video"])), {})
            if (saved.get("judgment_json") is not None and not saved.get("error")
                    and saved.get("judge_model") == args.judge_model
                    and saved.get("judge_prompt_version") == JUDGE_PROMPT_VERSION
                    and str(saved.get("description_prompt_version")) == str(row.get("prompt_version"))
                    and saved.get("gold") == gold_evidence and saved.get("model_response") == row["response"]):
                judged_rows.append(saved)
                print("Reusing matching successful judgment.")
                continue
            error = None
            try:
                if args.dry_run_prompts:
                    judgment_raw = JUDGE_PROMPT.format(gold=gold_evidence, model_response=row["response"])
                else:
                    judgment_raw = judge_with_openai(
                        args.judge_model,
                        gold=gold_evidence,
                        model_response=row["response"],
                    )
            except Exception as exc:
                judgment_raw = ""
                error = repr(exc)

            score = None if args.dry_run_prompts else parse_score(judgment_raw)
            judgment_json = None if score is None else {"overall_score": score}
            if not args.dry_run_prompts and not error and score is None:
                error = "Judge did not return a 0-100 integer."

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
