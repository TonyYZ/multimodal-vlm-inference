from __future__ import annotations

import argparse
import csv
import math
import re
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = PROJECT_ROOT / "results" / "model_comparison" / "ratings_paired.csv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results" / "model_charts"

INFERENCE_ORDER = [
    ("Scalar implication", ("ScalarImplicature1", "ScalarImplicature2", "ScalarImplicature3")),
    ("Presupposition", ("Presupposition1", "Presupposition2")),
    ("Supplement", ("Supplement1", "Supplement2")),
    ("Homogeneity", ("Homogeneity1", "Homogeneity2")),
]
CONDITION_ORDER = {
    "Scalar implication": ("Positive", "Negative"),
    "Presupposition": ("Question", "None"),
    "Supplement": ("Unembedded",),
    "Homogeneity": ("Positive", "Negative"),
}
PREMISE_ORDER = ("TargetPremise", "ControlPremise")
STATEMENT_LABELS = ("Target", "Baseline")

TARGET_COLOR = "#99efb8"
BASELINE_COLOR = "#d9d9d9"
TARGET_DOT = "#69c98e"
BASELINE_DOT = "#909090"
MODEL_TITLE_SIZE = 11
EXPERIMENT_LABEL_SIZE = 10


def model_slug(model: str) -> str:
    slug = model.lower().replace("/", "__")
    slug = re.sub(r"[^a-z0-9_.-]+", "_", slug)
    return slug.strip("_")


def display_model_name(model: str) -> str:
    return model.split("/")[-1]


def load_rows(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def numeric(value: str) -> float | None:
    if value == "":
        return None
    try:
        return float(value)
    except ValueError:
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


def stderr(values: list[float]) -> float:
    if len(values) <= 1:
        return 0.0
    return float(np.std(values, ddof=1) / math.sqrt(len(values)))


def summarize(values: list[float]) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    return float(np.mean(values)), stderr(values)


def grouped_values(rows: list[dict]) -> dict[tuple[str, str, str, str], list[float]]:
    grouped: dict[tuple[str, str, str, str], list[float]] = defaultdict(list)
    for row in rows:
        group = inference_group(row["task_family"])
        if group == "Other":
            continue
        condition = condition_label(row)
        premise = row["premise"]
        for statement_label, score_field in (
            ("Target", "target_score"),
            ("Baseline", "baseline_score"),
        ):
            score = numeric(row[score_field])
            if score is not None:
                grouped[(group, condition, premise, statement_label)].append(score)
    return grouped


def group_spec() -> list[dict]:
    specs = []
    for group, _ in INFERENCE_ORDER:
        for condition in CONDITION_ORDER[group]:
            premises = ["TargetPremise", "ControlPremise"]
            # Homogeneity and Presupposition currently have only target-premise videos.
            if group in {"Homogeneity", "Presupposition"}:
                premises = ["TargetPremise"]
            for premise in premises:
                specs.append(
                    {
                        "group": group,
                        "condition": condition,
                        "premise": premise,
                    }
                )
    return specs


def plot_model(model: str, rows: list[dict], output_dir: Path, formats: tuple[str, ...]) -> None:
    numeric_rows = [
        row
        for row in rows
        if numeric(row["target_score"]) is not None or numeric(row["baseline_score"]) is not None
    ]
    if not numeric_rows:
        print(f"Skipped {model}: no numeric scores found")
        return

    specs = group_spec()
    grouped = grouped_values(numeric_rows)

    x = np.arange(len(specs), dtype=float)
    width = 0.34
    offsets = {"Target": -width / 2, "Baseline": width / 2}
    colors = {"Target": TARGET_COLOR, "Baseline": BASELINE_COLOR}
    dot_colors = {"Target": TARGET_DOT, "Baseline": BASELINE_DOT}

    fig, ax = plt.subplots(figsize=(16, 5.2), dpi=180)

    for statement_label in STATEMENT_LABELS:
        means = []
        errors = []
        for spec in specs:
            vals = grouped[
                (
                    spec["group"],
                    spec["condition"],
                    spec["premise"],
                    statement_label,
                )
            ]
            mean, err = summarize(vals)
            means.append(mean)
            errors.append(err)

        xpos = x + offsets[statement_label]
        ax.bar(
            xpos,
            means,
            width=width,
            color=colors[statement_label],
            edgecolor="none",
            alpha=0.82,
            label=statement_label,
            zorder=2,
        )
        ax.errorbar(
            xpos,
            means,
            yerr=errors,
            fmt="none",
            ecolor="black",
            elinewidth=1.2,
            capsize=4,
            capthick=1.2,
            zorder=4,
        )

        rng = np.random.default_rng(24)
        for i, spec in enumerate(specs):
            vals = grouped[
                (
                    spec["group"],
                    spec["condition"],
                    spec["premise"],
                    statement_label,
                )
            ]
            if not vals:
                continue
            jitter = rng.normal(0, 0.035, size=len(vals))
            ax.scatter(
                np.full(len(vals), xpos[i]) + jitter,
                vals,
                s=16,
                color=dot_colors[statement_label],
                alpha=0.45,
                linewidths=0,
                zorder=3,
            )

    ax.set_ylim(-4, 104)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_yticklabels(["0%", "25%", "50%", "75%", "100%"])
    ax.set_ylabel("Mean endorsement", fontsize=13)
    ax.set_xlabel("Premises", fontsize=13)
    fig.suptitle(display_model_name(model), fontsize=MODEL_TITLE_SIZE, y=0.975)
    ax.grid(axis="y", color="#e9e9e9", linewidth=0.8)
    ax.set_axisbelow(True)

    ax.set_xticks(x)
    ax.set_xticklabels(
        ["Target" if spec["premise"] == "TargetPremise" else "Control" for spec in specs],
        fontsize=10,
    )

    add_group_headers(ax, specs)
    ax.legend(
        title="Inferences",
        frameon=False,
        bbox_to_anchor=(1.01, 0.62),
        loc="center left",
    )

    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    fig.text(
        0.015,
        0.47,
        "Animation\nExperiment",
        ha="left",
        va="center",
        fontsize=EXPERIMENT_LABEL_SIZE,
        linespacing=1.1,
    )
    fig.subplots_adjust(left=0.17, right=0.88, top=0.70, bottom=0.19)

    output_dir.mkdir(parents=True, exist_ok=True)
    for fmt in formats:
        path = output_dir / f"{model_slug(model)}_bar_chart.{fmt}"
        fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def add_group_headers(ax, specs: list[dict]) -> None:
    transform = ax.get_xaxis_transform()

    spans = []
    start = 0
    while start < len(specs):
        group = specs[start]["group"]
        end = start
        while end + 1 < len(specs) and specs[end + 1]["group"] == group:
            end += 1
        spans.append((group, start, end))
        start = end + 1

    condition_spans = []
    start = 0
    while start < len(specs):
        group = specs[start]["group"]
        condition = specs[start]["condition"]
        end = start
        while (
            end + 1 < len(specs)
            and specs[end + 1]["group"] == group
            and specs[end + 1]["condition"] == condition
        ):
            end += 1
        condition_spans.append((condition, start, end))
        start = end + 1

    for group, start, end in spans:
        x0 = start - 0.5
        x1 = end + 0.5
        ax.plot([x0, x1], [1.22, 1.22], color="black", linewidth=1.2, transform=transform, clip_on=False)
        ax.plot([x0, x0], [1.02, 1.22], color="black", linewidth=1.2, transform=transform, clip_on=False)
        ax.plot([x1, x1], [1.02, 1.22], color="black", linewidth=1.2, transform=transform, clip_on=False)
        ax.text((start + end) / 2, 1.17, group, ha="center", va="center", fontsize=14, transform=transform)

    for condition, start, end in condition_spans:
        x0 = start - 0.5
        x1 = end + 0.5
        rect = plt.Rectangle(
            (x0, 1.00),
            x1 - x0,
            0.095,
            transform=transform,
            facecolor="#e5e5e5",
            edgecolor="#a5a5a5",
            linewidth=0.8,
            clip_on=False,
        )
        ax.add_patch(rect)
        ax.text((start + end) / 2, 1.048, condition, ha="center", va="center", fontsize=10, transform=transform)

    for _, _, end in spans[:-1]:
        ax.axvline(end + 0.5, color="black", linewidth=1.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create grouped bar charts of target/baseline endorsement for each model."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--formats",
        nargs="+",
        default=["png", "pdf"],
        choices=["png", "pdf", "svg"],
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = load_rows(args.input)
    if not rows:
        raise ValueError(f"No rows found in {args.input}")

    by_model: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_model[row["model"]].append(row)

    for model, model_rows in sorted(by_model.items()):
        plot_model(model, model_rows, args.output_dir, tuple(args.formats))

    print(f"Wrote charts to {args.output_dir}")


if __name__ == "__main__":
    main()
