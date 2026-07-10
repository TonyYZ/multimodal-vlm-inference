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
DEFAULT_HUMAN_EFFECTS = PROJECT_ROOT / "materials" / "stats" / "TSC_EXP_2_human_effects.csv"

STATEMENT_TYPES = ("target", "baseline")
INFERENCE_ORDER = [
    ("Scalar implication", ("ScalarImplicature1", "ScalarImplicature2", "ScalarImplicature3")),
    ("Presupposition", ("Presupposition1", "Presupposition2")),
    ("Supplement", ("Supplement1", "Supplement2")),
    ("Homogeneity", ("Homogeneity1", "Homogeneity2")),
]
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
    raw_model: str
    input_mode: str
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
    if 0 <= score <= 100:
        return score
    return None


def inference_group(task_family: str) -> str:
    for label, prefixes in INFERENCE_ORDER:
        if task_family.startswith(prefixes):
            return label
    return "Other"


def condition_label(row: dict) -> str:
    if inference_group(row["task_family"]) == "Supplement":
        return "Unembedded"
    return row["condition"]


def human_condition_label(inference_group_label: str) -> str:
    if inference_group_label == "Scalar implication":
        return "Scalar Implicature"
    return inference_group_label


def numeric(value: object) -> float | None:
    if value in ("", None, "NA"):
        return None
    try:
        return float(value)
    except ValueError:
        return None


def model_label(model: str, input_mode: str) -> str:
    if not input_mode or input_mode == "split":
        return model
    return f"{model} [{input_mode}]"


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
                raw_model = str(raw.get("model", ""))
                input_mode = str(raw.get("input_mode", "split"))
                records.append(
                    RatingRecord(
                        model=model_label(raw_model, input_mode),
                        raw_model=raw_model,
                        input_mode=input_mode,
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


def load_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def long_rows(records: list[RatingRecord]) -> list[dict]:
    return [
        {
            "model": record.model,
            "raw_model": record.raw_model,
            "input_mode": record.input_mode,
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
                "raw_model": representative.raw_model,
                "input_mode": representative.input_mode,
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
            "input_mode": record.input_mode,
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
        "input_mode",
        *models,
    ]


def load_human_effects(path: Path) -> dict[tuple[str, str], float]:
    rows = load_csv(path)
    effects = {}
    for row in rows:
        human_effect = numeric(row.get("HUMAN_EFFECT"))
        if human_effect is None:
            continue
        effects[(str(row.get("Condition", "")).strip(), str(row.get("Environment", "")).strip())] = (
            human_effect
        )
    return effects


def model_human_effect_rows(
    paired: list[dict],
    human_effects: dict[tuple[str, str], float],
) -> list[dict]:
    deltas: dict[tuple[str, str, str, str, str], list[float]] = defaultdict(list)
    for row in paired:
        delta = numeric(row["target_minus_baseline"])
        if delta is None:
            continue
        group = inference_group(row["task_family"])
        if group == "Other":
            continue
        condition = condition_label(row)
        deltas[
            (
                row["raw_model"],
                row["input_mode"],
                group,
                condition,
                row["premise"],
            )
        ].append(delta)

    model_conditions = sorted({key[:4] for key in deltas})
    rows = []
    for raw_model, input_mode, group, condition in model_conditions:
        target_values = deltas[(raw_model, input_mode, group, condition, "TargetPremise")]
        control_values = deltas[(raw_model, input_mode, group, condition, "ControlPremise")]
        if not target_values:
            continue

        target_mean = mean(target_values)
        control_mean = mean(control_values) if control_values else None
        if group in {"Scalar implication", "Supplement"}:
            if control_mean is None:
                continue
            model_effect = target_mean - control_mean
            effect_type = "target_minus_control_delta"
        else:
            model_effect = target_mean
            effect_type = "target_delta"

        human_condition = human_condition_label(group)
        human_effect = human_effects.get((human_condition, condition))
        if human_effect is None:
            continue

        rows.append(
            {
                "raw_model": raw_model,
                "input_mode": input_mode,
                "inference_group": group,
                "condition": condition,
                "human_condition": human_condition,
                "human_environment": condition,
                "effect_type": effect_type,
                "target_premise_mean_delta": f"{target_mean:.3f}",
                "control_premise_mean_delta": f"{control_mean:.3f}" if control_mean is not None else "",
                "model_effect": f"{model_effect:.3f}",
                "human_effect": f"{human_effect:.3f}",
                "absolute_error": f"{abs(model_effect - human_effect):.3f}",
            }
        )

    return rows


def model_human_summary_rows(effect_rows: list[dict]) -> list[dict]:
    grouped: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in effect_rows:
        error = numeric(row["absolute_error"])
        if error is not None:
            grouped[(row["raw_model"], row["input_mode"])].append(error)

    rows = []
    for (raw_model, input_mode), errors in sorted(grouped.items()):
        rows.append(
            {
                "raw_model": raw_model,
                "input_mode": input_mode,
                "n_conditions": len(errors),
                "mean_absolute_error": f"{mean(errors):.3f}",
            }
        )
    return rows


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
    parser.add_argument(
        "--human-effects",
        nargs="?",
        const=DEFAULT_HUMAN_EFFECTS,
        default=None,
        type=Path,
        help=(
            "Optional human effects CSV. If passed without a path, defaults to "
            "materials/stats/TSC_EXP_2_human_effects.csv."
        ),
    )
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
            "raw_model",
            "input_mode",
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
            "raw_model",
            "input_mode",
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

    if args.human_effects is not None:
        human_effects = load_human_effects(args.human_effects)
        human_rows = model_human_effect_rows(paired, human_effects)
        human_summary = model_human_summary_rows(human_rows)
        write_csv(
            args.output_dir / "model_human_effects.csv",
            human_rows,
            [
                "raw_model",
                "input_mode",
                "inference_group",
                "condition",
                "human_condition",
                "human_environment",
                "effect_type",
                "target_premise_mean_delta",
                "control_premise_mean_delta",
                "model_effect",
                "human_effect",
                "absolute_error",
            ],
        )
        write_csv(
            args.output_dir / "model_human_summary.csv",
            human_summary,
            [
                "raw_model",
                "input_mode",
                "n_conditions",
                "mean_absolute_error",
            ],
        )

    print(f"Read {len(records)} rating records from {len(input_paths)} files.")
    print(f"Found {len(summary)} models.")
    print(f"Wrote comparison CSVs to {args.output_dir}")
    if args.human_effects is not None:
        print(f"Wrote human-effect comparison CSVs using {args.human_effects}")


if __name__ == "__main__":
    main()
