from __future__ import annotations

import argparse
import csv
import math
import re
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = PROJECT_ROOT / "results" / "model_comparison" / "ratings_paired.csv"
DEFAULT_HUMAN = PROJECT_ROOT / "materials" / "stats" / "TSC_EXP_2_human_effects.csv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results" / "model_charts"

INFERENCE_ORDER = [
    ("Scalar implicature", ("ScalarImplicature1", "ScalarImplicature2", "ScalarImplicature3")),
    ("Presupposition", ("Presupposition1", "Presupposition2")),
    ("Supplement", ("Supplement1", "Supplement2")),
    ("Homogeneity", ("Homogeneity1", "Homogeneity2")),
]
CONDITION_ORDER = {
    "Scalar implicature": ("Positive", "Negative"),
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
COMBINED_PAGE_SIZE = (11.69, 12.5)
DEFAULT_MODELS_PER_PAGE = 3


def model_slug(model: str) -> str:
    slug = model.lower().replace("/", "__")
    slug = re.sub(r"[^a-z0-9_.-]+", "_", slug)
    return slug.strip("_")


def display_model_name(model: str) -> str:
    return model.split("/")[-1]


def model_sort_key(model: str) -> tuple[int, float, str]:
    name = model.lower()
    size_match = re.search(r"(\d+(?:\.\d+)?)b", name)
    size = float(size_match.group(1)) if size_match else 999.0
    if "qwen2.5-vl" in name:
        return (0, size, model)
    if "qwen3-vl" in name:
        return (1, size, model)
    if "qwen3.8" in name:
        return (2, size, model)
    if "gemma" in name:
        return (3, size, model)
    return (4, size, model)


def load_rows(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def save_figure(fig, path: Path, tight: bool = True) -> None:
    try:
        if tight:
            fig.savefig(path, bbox_inches="tight")
        else:
            fig.savefig(path)
    except OSError as exc:
        print(f"Warning: could not write {path}: {exc}")


def save_pdf_page(pdf: PdfPages, fig, tight: bool = True) -> None:
    try:
        if tight:
            pdf.savefig(fig, bbox_inches="tight")
        else:
            pdf.savefig(fig)
    except OSError as exc:
        print(f"Warning: could not write PDF page: {exc}")


def human_rows(path: Path) -> list[dict]:
    task_prefix = {
        "Scalar Implicature": "ScalarImplicature1",
        "Presupposition": "Presupposition1",
        "Supplement": "Supplement1",
        "Homogeneity": "Homogeneity1",
    }
    rows = []
    for row in load_rows(path):
        group = row["Condition"]
        condition = row["Environment"]
        for premise, target_col, baseline_col in (
            ("TargetPremise", "Target_Target", "Target_Baseline"),
            ("ControlPremise", "Control_Target", "Control_Baseline"),
        ):
            target_score = row.get(target_col, "")
            baseline_score = row.get(baseline_col, "")
            if target_score in {"", "NA"} and baseline_score in {"", "NA"}:
                continue
            rows.append(
                {
                    "raw_model": "Human",
                    "model": "Human",
                    "task_family": f"{task_prefix[group]}Human",
                    "condition": condition,
                    "premise": premise,
                    "target_score": "" if target_score == "NA" else target_score,
                    "baseline_score": "" if baseline_score == "NA" else baseline_score,
                }
            )
    return rows


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


def plot_bars(
    ax,
    rows: list[dict],
    title: str,
    show_xlabel: bool,
    show_ylabel: bool,
    show_legend: bool,
    compact_headers: bool,
) -> bool:
    numeric_rows = [
        row
        for row in rows
        if numeric(row["target_score"]) is not None or numeric(row["baseline_score"]) is not None
    ]
    if not numeric_rows:
        return False

    specs = group_spec()
    grouped = grouped_values(numeric_rows)

    x = np.arange(len(specs), dtype=float)
    width = 0.34
    offsets = {"Target": -width / 2, "Baseline": width / 2}
    colors = {"Target": TARGET_COLOR, "Baseline": BASELINE_COLOR}
    dot_colors = {"Target": TARGET_DOT, "Baseline": BASELINE_DOT}

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
    if show_ylabel:
        ax.set_ylabel("Mean endorsement", fontsize=12)
    if show_xlabel:
        ax.set_xlabel("Premises", fontsize=12)
    ax.set_title(title, fontsize=MODEL_TITLE_SIZE, loc="left", pad=58 if compact_headers else 34)
    ax.grid(axis="y", color="#e9e9e9", linewidth=0.8)
    ax.set_axisbelow(True)

    ax.set_xticks(x)
    ax.set_xticklabels(
        ["Target" if spec["premise"] == "TargetPremise" else "Control" for spec in specs],
        fontsize=10,
    )

    add_group_headers(ax, specs, compact=compact_headers)
    if show_legend:
        ax.legend(
            title="Inferences",
            frameon=False,
            bbox_to_anchor=(1.01, 0.62),
            loc="center left",
        )

    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    return True


def plot_model(model: str, rows: list[dict], output_dir: Path, formats: tuple[str, ...]) -> None:
    fig, ax = plt.subplots(figsize=(16, 5.2), dpi=180)
    plotted = plot_bars(
        ax,
        rows,
        title=display_model_name(model),
        show_xlabel=True,
        show_ylabel=True,
        show_legend=True,
        compact_headers=False,
    )
    if not plotted:
        print(f"Skipped {model}: no numeric scores found")
        plt.close(fig)
        return

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
        save_figure(fig, path)
    plt.close(fig)


def plot_combined_input_mode(
    input_mode: str,
    rows: list[dict],
    human_reference_rows: list[dict],
    output_dir: Path,
    formats: tuple[str, ...],
    models_per_page: int,
) -> None:
    by_raw_model: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_raw_model[row["raw_model"]].append(row)

    models = sorted(by_raw_model, key=model_sort_key)
    label = {
        "split": "Split",
        "pure-text": "Pure text",
        "pure-video": "Pure video",
    }.get(input_mode, input_mode)
    output_dir.mkdir(parents=True, exist_ok=True)
    mode_slug = input_mode.replace("-", "_")
    pdf = None
    if "pdf" in formats:
        try:
            pdf = PdfPages(output_dir / f"all_models_{mode_slug}_bar_charts.pdf")
        except OSError as exc:
            print(f"Warning: could not open combined PDF for {input_mode}: {exc}")

    try:
        for page_index, start in enumerate(range(0, len(models), models_per_page), start=1):
            page_models = models[start : start + models_per_page]
            panels = [("Human", human_reference_rows), *[(model, by_raw_model[model]) for model in page_models]]
            slots_per_page = models_per_page + 1
            fig, axes = plt.subplots(
                slots_per_page,
                1,
                figsize=COMBINED_PAGE_SIZE,
                dpi=180,
                squeeze=False,
            )
            for i, (model, panel_rows) in enumerate(panels):
                plot_bars(
                    axes[i, 0],
                    panel_rows,
                    title=display_model_name(model),
                    show_xlabel=i == len(panels) - 1,
                    show_ylabel=True,
                    show_legend=i == 0,
                    compact_headers=True,
                )
            for j in range(len(panels), slots_per_page):
                axes[j, 0].axis("off")
            fig.suptitle(
                f"{label}: model comparison, page {page_index}",
                fontsize=12,
                y=0.995,
            )
            fig.subplots_adjust(left=0.12, right=0.86, top=0.90, bottom=0.06, hspace=0.75)

            if pdf is not None:
                save_pdf_page(pdf, fig, tight=False)
            for fmt in formats:
                if fmt == "pdf":
                    continue
                save_figure(
                    fig,
                    output_dir / f"all_models_{mode_slug}_bar_charts_page{page_index}.{fmt}",
                    tight=False,
                )
            plt.close(fig)
    finally:
        if pdf is not None:
            pdf.close()


def add_group_headers(ax, specs: list[dict], compact: bool = False) -> None:
    transform = ax.get_xaxis_transform()
    group_font = 10 if compact else 14
    condition_font = 8 if compact else 10

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
        ax.text((start + end) / 2, 1.17, group, ha="center", va="center", fontsize=group_font, transform=transform)

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
        ax.text((start + end) / 2, 1.048, condition, ha="center", va="center", fontsize=condition_font, transform=transform)

    for _, _, end in spans[:-1]:
        ax.axvline(end + 0.5, color="black", linewidth=1.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create grouped bar charts of target/baseline endorsement for each model."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--human", type=Path, default=DEFAULT_HUMAN)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--formats",
        nargs="+",
        default=["png", "pdf"],
        choices=["png", "pdf", "svg"],
    )
    parser.add_argument(
        "--skip-combined",
        action="store_true",
        help="Only write the existing per-model charts.",
    )
    parser.add_argument(
        "--models-per-page",
        type=int,
        default=DEFAULT_MODELS_PER_PAGE,
        help="Number of model panels per A4 comparison page, excluding the repeated human panel.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = load_rows(args.input)
    if not rows:
        raise ValueError(f"No rows found in {args.input}")
    human_reference_rows = human_rows(args.human)

    by_model: dict[str, list[dict]] = defaultdict(list)
    by_input_mode: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_model[row["model"]].append(row)
        by_input_mode[row["input_mode"]].append(row)

    for model, model_rows in sorted(by_model.items()):
        plot_model(model, model_rows, args.output_dir, tuple(args.formats))

    if not args.skip_combined:
        for input_mode in ("split", "pure-text", "pure-video"):
            if input_mode in by_input_mode:
                plot_combined_input_mode(
                    input_mode,
                    by_input_mode[input_mode],
                    human_reference_rows,
                    args.output_dir,
                    tuple(args.formats),
                    args.models_per_page,
                )

    print(f"Wrote charts to {args.output_dir}")


if __name__ == "__main__":
    main()
