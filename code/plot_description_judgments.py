from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, median, stdev

import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_JUDGMENTS = PROJECT_ROOT / "results" / "description_judgments_all_models.jsonl"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results" / "description_judge_charts"

MODEL_ORDER = (
    "Qwen/Qwen2.5-VL-3B-Instruct",
    "Qwen/Qwen2.5-VL-7B-Instruct",
    "Qwen/Qwen2.5-VL-32B-Instruct",
    "Qwen/Qwen3-VL-2B-Instruct",
    "Qwen/Qwen3-VL-8B-Instruct",
    "Qwen/Qwen3.8-27B",
    "google/gemma-4-12B-it",
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
            grouped[(row["model"], row["input_mode"])].append(judgment)

    summary = []
    for (model, mode), judgments in grouped.items():
        scores = [float(row["overall_score"]) for row in judgments]
        visual_scores = [
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
        visual_sd = stdev(visual_scores) if len(visual_scores) > 1 else 0.0
        summary.append(
            {
                "model": model,
                "input_mode": mode,
                "n": len(scores),
                "mean_score": mean(scores),
                "median_score": median(scores),
                "standard_deviation": sd,
                "standard_error": sd / math.sqrt(len(scores)),
                "mean_visual_score": mean(visual_scores),
                "visual_standard_error": visual_sd / math.sqrt(len(visual_scores)),
                **{
                    f"mean_{field}": 50.0 * mean(float(row[field]) for row in judgments)
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
            "mean_visual_score": mean(row["mean_visual_score"] for row in model_rows),
        }
        for model, model_rows in by_model.items()
    ]
    for field, rank_field in (
        ("mean_overall_score", "overall_rank"),
        ("mean_visual_score", "visual_rank"),
    ):
        for rank, row in enumerate(
            sorted(rankings, key=lambda item: item[field], reverse=True), start=1
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

    fig, ax = plt.subplots(figsize=(11, 6.5))
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
    fig.tight_layout()

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
        default=[DEFAULT_JUDGMENTS],
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = summarize(
        [row for path in args.judgments for row in load_rows(path)]
    )
    if not summary:
        raise ValueError(f"No successful judgments found in {args.judgments}")
    write_summary(summary, args.output_dir / "description_judge_model_summary.csv")
    write_rankings(summary, args.output_dir / "description_judge_model_ranking.csv")
    plot(
        summary,
        args.output_dir,
        mean_field="mean_score",
        error_field="standard_error",
        title="VLM sequence-description accuracy",
        xlabel="Mean overall score",
        filename="description_judge_model_comparison",
    )
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
