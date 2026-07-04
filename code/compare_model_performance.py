from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import mean


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS_DIR = PROJECT_ROOT / "results"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results" / "model_comparison"

STATEMENT_TYPES = ("target", "baseline")
PREMISE_ORDER = {
    "TargetPremise": 0,
    "ControlPremise": 1,
}
CONDITION_ORDER = {
    "Positive": 0,
    "Question": 0,
    "Negative": 1,
    "None": 1,
}


@dataclass
class RatingRecord:
    model: str
    source_video: str
    task_family: str
    premise: str
    condition: str
    statement_type: str
    statement: str
    response: str
    score: int | None
    error: str
    result_file: str
    line_number: int


def parse_task_name(source_video: str) -> tuple[str, str, str]:
    task_name = Path(source_video).stem
    condition = ""
    condition_match = re.search(r"\(([^)]+)\)$", task_name)
    if condition_match:
        condition = condition_match.group(1)
        task_name = task_name[: condition_match.start()]

    premise = ""
    family = task_name
    for premise_name in PREMISE_ORDER:
        marker = f"_{premise_name}"
        if marker in task_name:
            family = task_name.replace(marker, "")
            premise = premise_name
            break

    return family, premise, condition


def task_sort_key(record: RatingRecord) -> tuple:
    return (
        record.task_family,
        CONDITION_ORDER.get(record.condition, 2),
        record.condition,
        PREMISE_ORDER.get(record.premise, 2),
        STATEMENT_TYPES.index(record.statement_type)
        if record.statement_type in STATEMENT_TYPES
        else len(STATEMENT_TYPES),
        record.model,
    )


def parse_score(response: object) -> int | None:
    if response is None:
        return None
    match = re.search(r"\b(?:100|[1-9][0-9]?|0)\b", str(response))
    if not match:
        return None
    score = int(match.group(0))
    if 1 <= score <= 100:
        return score
    return None


def discover_result_files(results_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in results_dir.glob("*sequence_ratings.jsonl")
        if path.is_file() and "__" in path.name
    )


def load_records(paths: list[Path]) -> list[RatingRecord]:
    records = []
    for path in paths:
        with path.open("r", encoding="utf-8") as f:
            for line_number, line in enumerate(f, start=1):
                if not line.strip():
                    continue
                raw = json.loads(line)
                source_video = str(raw.get("source_video", ""))
                task_family, premise, condition = parse_task_name(source_video)
                response = str(raw.get("response", ""))
                records.append(
                    RatingRecord(
                        model=str(raw.get("model", "")),
                        source_video=source_video,
                        task_family=task_family,
                        premise=premise,
                        condition=condition,
                        statement_type=str(raw.get("statement_type", "")),
                        statement=str(raw.get("statement", "")),
                        response=response,
                        score=parse_score(response),
                        error=str(raw.get("error", "")),
                        result_file=str(path.relative_to(PROJECT_ROOT)),
                        line_number=line_number,
                    )
                )
    return sorted(records, key=task_sort_key)


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def long_rows(records: list[RatingRecord]) -> list[dict]:
    return [
        {
            "model": record.model,
            "source_video": record.source_video,
            "task_family": record.task_family,
            "premise": record.premise,
            "condition": record.condition,
            "statement_type": record.statement_type,
            "statement": record.statement,
            "score": record.score if record.score is not None else "",
            "response": record.response,
            "error": record.error,
            "result_file": record.result_file,
            "line_number": record.line_number,
        }
        for record in records
    ]


def paired_rows(records: list[RatingRecord]) -> list[dict]:
    grouped: dict[tuple[str, str], dict[str, RatingRecord]] = defaultdict(dict)
    for record in records:
        grouped[(record.model, record.source_video)][record.statement_type] = record

    rows = []
    for (model, source_video), by_type in grouped.items():
        target = by_type.get("target")
        baseline = by_type.get("baseline")
        representative = target or baseline
        if representative is None:
            continue
        target_score = target.score if target else None
        baseline_score = baseline.score if baseline else None
        delta = (
            target_score - baseline_score
            if target_score is not None and baseline_score is not None
            else None
        )
        rows.append(
            {
                "model": model,
                "source_video": source_video,
                "task_family": representative.task_family,
                "premise": representative.premise,
                "condition": representative.condition,
                "target_score": target_score if target_score is not None else "",
                "baseline_score": baseline_score if baseline_score is not None else "",
                "target_minus_baseline": delta if delta is not None else "",
                "target_response": target.response if target else "",
                "baseline_response": baseline.response if baseline else "",
            }
        )
    return sorted(
        rows,
        key=lambda row: (
            row["task_family"],
            CONDITION_ORDER.get(row["condition"], 2),
            row["condition"],
            PREMISE_ORDER.get(row["premise"], 2),
            row["model"],
        ),
    )


def summary_rows(records: list[RatingRecord], paired: list[dict]) -> list[dict]:
    by_model: dict[str, list[RatingRecord]] = defaultdict(list)
    paired_by_model: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        by_model[record.model].append(record)
    for row in paired:
        paired_by_model[row["model"]].append(row)

    rows = []
    for model in sorted(by_model):
        model_records = by_model[model]
        numeric = [record.score for record in model_records if record.score is not None]
        target_scores = [
            record.score
            for record in model_records
            if record.statement_type == "target" and record.score is not None
        ]
        baseline_scores = [
            record.score
            for record in model_records
            if record.statement_type == "baseline" and record.score is not None
        ]
        deltas = [
            int(row["target_minus_baseline"])
            for row in paired_by_model[model]
            if row["target_minus_baseline"] != ""
        ]
        rows.append(
            {
                "model": model,
                "n_records": len(model_records),
                "n_numeric": len(numeric),
                "n_errors": sum(1 for record in model_records if record.error),
                "mean_score": f"{mean(numeric):.3f}" if numeric else "",
                "mean_target_score": f"{mean(target_scores):.3f}" if target_scores else "",
                "mean_baseline_score": f"{mean(baseline_scores):.3f}"
                if baseline_scores
                else "",
                "mean_target_minus_baseline": f"{mean(deltas):.3f}" if deltas else "",
            }
        )
    return rows


def wide_rows(records: list[RatingRecord]) -> tuple[list[dict], list[str]]:
    grouped: dict[tuple[str, str, str], dict[str, str]] = defaultdict(dict)
    models = sorted({record.model for record in records})

    metadata: dict[tuple[str, str, str], RatingRecord] = {}
    for record in records:
        key = (record.source_video, record.statement_type, record.statement)
        metadata[key] = record
        grouped[key][record.model] = (
            str(record.score) if record.score is not None else record.response
        )

    rows = []
    for key, model_scores in grouped.items():
        record = metadata[key]
        row = {
            "source_video": record.source_video,
            "task_family": record.task_family,
            "premise": record.premise,
            "condition": record.condition,
            "statement_type": record.statement_type,
            "statement": record.statement,
        }
        for model in models:
            row[model] = model_scores.get(model, "")
        rows.append(row)

    rows.sort(
        key=lambda row: (
            row["task_family"],
            CONDITION_ORDER.get(row["condition"], 2),
            row["condition"],
            PREMISE_ORDER.get(row["premise"], 2),
            STATEMENT_TYPES.index(row["statement_type"])
            if row["statement_type"] in STATEMENT_TYPES
            else len(STATEMENT_TYPES),
        )
    )
    return rows, [
        "source_video",
        "task_family",
        "premise",
        "condition",
        "statement_type",
        "statement",
        *models,
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare model rating outputs written by feed_experiment.py."
    )
    parser.add_argument(
        "inputs",
        nargs="*",
        type=Path,
        help="Result JSONL files. Defaults to results/*sequence_ratings.jsonl.",
    )
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_paths = args.inputs or discover_result_files(args.results_dir)
    if not input_paths:
        raise FileNotFoundError(
            f"No result files found. Expected files like {args.results_dir}/*sequence_ratings.jsonl"
        )

    records = load_records(input_paths)
    paired = paired_rows(records)
    summary = summary_rows(records, paired)
    wide, wide_fieldnames = wide_rows(records)

    write_csv(
        args.output_dir / "ratings_long.csv",
        long_rows(records),
        [
            "model",
            "source_video",
            "task_family",
            "premise",
            "condition",
            "statement_type",
            "statement",
            "score",
            "response",
            "error",
            "result_file",
            "line_number",
        ],
    )
    write_csv(
        args.output_dir / "ratings_paired.csv",
        paired,
        [
            "model",
            "source_video",
            "task_family",
            "premise",
            "condition",
            "target_score",
            "baseline_score",
            "target_minus_baseline",
            "target_response",
            "baseline_response",
        ],
    )
    write_csv(
        args.output_dir / "model_summary.csv",
        summary,
        [
            "model",
            "n_records",
            "n_numeric",
            "n_errors",
            "mean_score",
            "mean_target_score",
            "mean_baseline_score",
            "mean_target_minus_baseline",
        ],
    )
    write_csv(args.output_dir / "ratings_wide.csv", wide, wide_fieldnames)

    print(f"Read {len(records)} rating records from {len(input_paths)} files.")
    print(f"Found {len(summary)} models.")
    print(f"Wrote comparison CSVs to {args.output_dir}")


if __name__ == "__main__":
    main()
