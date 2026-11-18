from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, median, stdev

import matplotlib.pyplot as plt
import numpy as np

from compare_model_performance import parse_task_name, inference_group
from build_significant_difference_profiles import TEST_ORDER
from judge_descriptions import extract_json


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_JUDGMENTS = PROJECT_ROOT / "results" / "animation" / "perception" / "judgments" / "description_judgments_all_models.jsonl"
V4_JUDGMENTS = DEFAULT_JUDGMENTS.with_name("description_judgments_v4_all_models.jsonl")
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results" / "animation" / "perception" / "charts"

MODEL_ORDER = (
    "Qwen/Qwen2.5-VL-3B-Instruct",
    "Qwen/Qwen2.5-VL-7B-Instruct",
    "Qwen/Qwen2.5-VL-32B-Instruct",
    "Qwen/Qwen3-VL-2B-Instruct",
    "Qwen/Qwen3-VL-8B-Instruct",
    "Qwen/Qwen3.8-27B",
    "google/gemma-4-12B-it",
    "Qwen/Qwen2.5-Omni-3B",
    "Qwen/Qwen2.5-Omni-7B",
    "Qwen/Qwen3-Omni-30B-A3B-Instruct",
    "openbmb/MiniCPM-o-4_5",
)
MODEL_LABELS = {
    model: model.replace("Qwen/Qwen", "Qwen").replace("-Instruct", "").replace("google/", "")
    for model in MODEL_ORDER
}
MODE_ORDER = ("pure-text", "pure-video", "split")
MODE_COLORS = {
    "pure-text": "#8c8c8c",
    "pure-video": "#4c78a8",
    "split": "#62c98d",
}
VISUAL_FIELDS = ("object_score", "spatial_score", "temporal_score", "event_type_score")


def load_rows(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def summarize(rows: list[dict]) -> list[dict]:
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        judgment = row.get("judgment_json") or {}
        score = judgment.get("overall_score")
        if isinstance(score, (int, float)) and "error" not in row:
            grouped[(row["model"], row["input_mode"])].append(row)

    summary = []
    for (model, mode), records in grouped.items():
        judgments = [row["judgment_json"] for row in records]
        scores = [float(row["overall_score"]) for row in judgments]
        # Judge v5 returns one match score; the per-dimension (visual-only) scores exist only for v4 judgments.
        rubric = all(field in row for row in judgments for field in (*VISUAL_FIELDS, "text_score"))
        visual_scores = [] if not rubric else [
            max(
                0.0,
                min(
                    100.0,
                    12.5 * sum(float(row[field]) for field in VISUAL_FIELDS)
                    - 5.0 * float(row["hallucination_penalty"]),
                ),
            )
            for row in judgments
        ]
        sd = stdev(scores) if len(scores) > 1 else 0.0
        visual_sd = stdev(visual_scores) if len(visual_scores) > 1 else 0.0 if rubric else None
        summary.append(
            {
                "model": model,
                "input_mode": mode,
                "description_prompt_versions": ",".join(sorted({str(r.get("description_prompt_version", "unknown")) for r in records})),
                "judge_prompt_versions": ",".join(sorted({str(r.get("judge_prompt_version", "unknown")) for r in records})),
                "n": len(scores),
                "mean_score": mean(scores),
                "median_score": median(scores),
                "standard_deviation": sd,
                "standard_error": sd / math.sqrt(len(scores)),
                "mean_visual_score": mean(visual_scores) if rubric else None,
                "visual_standard_error": visual_sd / math.sqrt(len(visual_scores)) if rubric else None,
                **{
                    f"mean_{field}": 50.0 * mean(float(row[field]) for row in judgments) if rubric else None
                    for field in ("text_score", *VISUAL_FIELDS)
                },
            }
        )
    return summary


def sort_key(row: dict) -> tuple[int, int]:
    model = row["model"]
    mode = row["input_mode"]
    return (
        MODEL_ORDER.index(model) if model in MODEL_ORDER else len(MODEL_ORDER),
        MODE_ORDER.index(mode) if mode in MODE_ORDER else len(MODE_ORDER),
    )


def write_summary(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = (
        "model",
        "input_mode",
        "description_prompt_versions",
        "judge_prompt_versions",
        "n",
        "mean_score",
        "median_score",
        "standard_deviation",
        "standard_error",
        "mean_visual_score",
        "visual_standard_error",
        "mean_text_score",
        "mean_object_score",
        "mean_spatial_score",
        "mean_temporal_score",
        "mean_event_type_score",
    )
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(sorted(rows, key=sort_key))


def write_rankings(rows: list[dict], path: Path) -> None:
    by_model: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_model[row["model"]].append(row)

    rankings = [
        {
            "model": model,
            "n_input_modes": len(model_rows),
            "mean_overall_score": mean(row["mean_score"] for row in model_rows),
            "mean_visual_score": None if any(row["mean_visual_score"] is None for row in model_rows)
            else mean(row["mean_visual_score"] for row in model_rows),
        }
        for model, model_rows in by_model.items()
    ]
    for field, rank_field in (
        ("mean_overall_score", "overall_rank"),
        ("mean_visual_score", "visual_rank"),
    ):
        ranked = [row for row in rankings if row[field] is not None]
        for rank, row in enumerate(
            sorted(ranked, key=lambda item: item[field], reverse=True), start=1
        ):
            row[rank_field] = rank

    fields = (
        "model",
        "n_input_modes",
        "mean_overall_score",
        "overall_rank",
        "mean_visual_score",
        "visual_rank",
    )
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(sorted(rankings, key=lambda row: row["overall_rank"]))


def plot(
    rows: list[dict],
    output_dir: Path,
    *,
    mean_field: str,
    error_field: str,
    title: str,
    xlabel: str,
    filename: str,
) -> None:
    model_scores = {
        model: mean(row[mean_field] for row in rows if row["model"] == model)
        for model in {row["model"] for row in rows}
    }
    models = sorted(model_scores, key=model_scores.get, reverse=True)
    modes = [mode for mode in MODE_ORDER if any(r["input_mode"] == mode for r in rows)]
    lookup = {(r["model"], r["input_mode"]): r for r in rows}

    fig, ax = plt.subplots(figsize=(12, max(6.5, .5 * len(models) + 1.5)))
    height = 0.72 / max(1, len(modes))
    centers = list(range(len(models)))
    for index, mode in enumerate(modes):
        offset = (index - (len(modes) - 1) / 2) * height
        present = [(i, lookup.get((model, mode))) for i, model in enumerate(models)]
        present = [(i, row) for i, row in present if row]
        y = [i + offset for i, _ in present]
        means = [row[mean_field] for _, row in present]
        errors = [row[error_field] for _, row in present]
        bars = ax.barh(
            y,
            means,
            height,
            xerr=errors,
            capsize=3,
            color=MODE_COLORS.get(mode, "#999999"),
            label=mode,
        )
        ax.bar_label(bars, labels=[f"{value:.1f}" for value in means], padding=3, fontsize=8)

    ax.set_yticks(centers, [MODEL_LABELS.get(model, model) for model in models])
    ax.invert_yaxis()
    ax.set_xlabel(xlabel)
    ax.set_xlim(0, 105)
    ax.set_title(title)
    ax.grid(axis="x", alpha=0.25)
    ax.legend(title="Input format", frameon=False)
    versions = sorted({v for row in rows for v in row.get("description_prompt_versions", "").split(",") if v})
    if len(versions) > 1:
        fig.text(.5, .015, "Description prompt versions: " + ", ".join(versions)
                 + "; model-specific versions are recorded in the score table. Judge rubric unchanged.",
                 ha="center", fontsize=8)
    fig.tight_layout(rect=(0, .04 if len(versions) > 1 else 0, 1, 1))

    output_dir.mkdir(parents=True, exist_ok=True)
    for extension in ("png", "pdf"):
        fig.savefig(output_dir / f"{filename}.{extension}", dpi=200)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare VLM description judgments.")
    parser.add_argument(
        "--judgments",
        type=Path,
        nargs="+",
        default=[V4_JUDGMENTS] if V4_JUDGMENTS.exists() else [
            DEFAULT_JUDGMENTS,
            DEFAULT_JUDGMENTS.with_name("description_judgments_pure_video_all_models.jsonl"),
            *sorted(DEFAULT_JUDGMENTS.parent.glob("description_judgments_added_models_*.jsonl")),
        ],
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def per_test_outputs(rows: list[dict], output: Path) -> None:
    groups = defaultdict(list)
    audit = []
    for row in rows:
        family, premise, condition = parse_task_name(row["source_video"])
        test = f"{inference_group(family)}: {condition or 'Unembedded'}"
        groups[test].append(row)
        parsed = extract_json(row.get("model_response", ""))
        timeline = (parsed or {}).get("timeline", [])
        descriptions = [e.get("description", "") for e in timeline if e.get("type") == "visual"]
        judgment = row.get("judgment_json") or {}
        audit.append({"model": row["model"], "input_mode": row["input_mode"],
                      "source_video": row["source_video"], "test": test, "premise": premise,
                      "valid_description_json": parsed is not None,
                      "n_visual_entries": len(descriptions),
                      "repeated_visual_descriptions": len(descriptions) - len(set(descriptions)),
                      **{field: judgment.get(field) for field in ("text_score", *VISUAL_FIELDS)},
                      "hallucination_penalty": judgment.get("hallucination_penalty"),
                      "major_errors": json.dumps(judgment.get("major_errors", []), ensure_ascii=False)})
    summary = [{"test": test, **r} for test, group in groups.items() for r in summarize(group)]
    for name, records in (("description_judge_by_test.csv", summary), ("description_response_audit.csv", audit)):
        with (output / name).open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)
    modes = [mode for mode in MODE_ORDER if any(r["input_mode"] == mode for r in summary)]
    found = {row["model"] for row in summary}
    models = [model for model in MODEL_ORDER if model in found] + sorted(found - set(MODEL_ORDER))
    tests = [test for test in TEST_ORDER if test in groups]
    fig, axes = plt.subplots(1, len(modes), figsize=(8 * len(modes), 6), squeeze=False, layout="constrained")
    rubric = all(r["mean_visual_score"] is not None for r in summary)
    field = "mean_visual_score" if rubric else "mean_score"
    lookup = {(r["model"], r["input_mode"], r["test"]): r[field] for r in summary}
    for ax, mode in zip(axes[0], modes):
        matrix = np.array([[lookup.get((model, mode, test), np.nan) for test in tests] for model in models])
        image = ax.imshow(matrix, vmin=0, vmax=100, cmap="YlGnBu", aspect="auto")
        ax.set_xticks(range(len(tests)), tests, rotation=40, ha="right", fontsize=8)
        ax.set_yticks(range(len(models)), [MODEL_LABELS.get(m, m) for m in models], fontsize=9)
        ax.set_title(mode)
        for y, values in enumerate(matrix):
            for x, value in enumerate(values):
                ax.text(x, y, f"{value:.1f}", ha="center", va="center", fontsize=8,
                        color="white" if value > 65 else "black")
    fig.colorbar(image, ax=list(axes[0]), label=f"Mean {'visual-only ' if rubric else ''}judge score (0-100)", shrink=.7)
    fig.suptitle("Perceptual accuracy by inference type and environment; descriptive judge scores", fontsize=12)
    for ext in ("png", "pdf"):
        fig.savefig(output / f"description_judge_visual_by_test.{ext}", dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    rows = [row for path in args.judgments for row in load_rows(path)]
    # Prevent repeated file arguments or appended retries from inflating sample sizes.
    rows = list({(r["model"], r["input_mode"], r["source_video"]): r for r in rows}.values())
    summary = summarize(rows)
    if not summary:
        raise ValueError(f"No successful judgments found in {args.judgments}")
    write_summary(summary, args.output_dir / "description_judge_model_summary.csv")
    write_rankings(summary, args.output_dir / "description_judge_model_ranking.csv")
    per_test_outputs(rows, args.output_dir)
    plot(
        summary,
        args.output_dir,
        mean_field="mean_score",
        error_field="standard_error",
        title="VLM sequence-description accuracy",
        xlabel="Mean overall score",
        filename="description_judge_model_comparison",
    )
    if any(row["mean_visual_score"] is None for row in summary):
        print(f"Wrote comparison outputs to {args.output_dir} (no visual-only scores in judge v5)")
        return
    plot(
        summary,
        args.output_dir,
        mean_field="mean_visual_score",
        error_field="visual_standard_error",
        title="VLM animation-description accuracy",
        xlabel="Mean visual-only score",
        filename="description_judge_visual_comparison",
    )
    print(f"Wrote comparison outputs to {args.output_dir}")


if __name__ == "__main__":
    main()
