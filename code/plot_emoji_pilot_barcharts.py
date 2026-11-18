from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from scipy.spatial.distance import pdist, squareform

from plot_model_barcharts import BASELINE_COLOR, TARGET_COLOR


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = PROJECT_ROOT / "results" / "emoji" / "inference" / "comparison" / "emoji_pilot_model_human.csv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results" / "emoji" / "inference" / "bar_charts"
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot matched human and model endorsement bars for the emoji pilot."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--models-per-page", type=int, default=3)
    return parser.parse_args()


def load_rows(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def experiment_spec(experiment: str) -> tuple[list[tuple[str, str]], str, str]:
    if experiment == "Experiment 1":
        cells = [
            ("Each", "Existential"),
            ("Each", "Universal"),
            ("Exactly-one", "Existential"),
            ("Exactly-one", "Universal"),
            ("Might", "Conditional"),
            ("Negation", "Conditional"),
            ("None", "Existential"),
            ("None", "Universal"),
            ("Unembedded", "Manner"),
        ]
        return cells, "Co-text emoji", "Material condition"
    if experiment == "Experiment 2":
        return [
            (environment, "")
            for environment in ("Unembedded", "Question", "Might", "Negation", "None")
        ], "Pro-text emoji", "Inference"
    return [
        (emoji, "")
        for emoji in ("Cigarette", "Umbrella", "Phone", "Plane", "Toilet")
    ], "Post-text emoji", "Inference"


def cell_key(row: dict) -> tuple[str, str]:
    if row["experiment"] == "Experiment 1":
        return row["environment"], row["inference_type"]
    if row["experiment"] == "Experiment 2":
        return row["environment"], ""
    return row["emoji"], ""


def condition_key(row: dict) -> str:
    if row["experiment"] == "Experiment 1":
        return row["material_condition"]
    return "Target" if row["statement_type"] == "target" else "Control"


def label(cell: tuple[str, str]) -> str:
    first, second = cell
    first = first.replace("Exactly-one", "Exactly one")
    return f"{first}\n{second}" if second not in {"", "Conditional", "Manner"} else first


def values(rows: list[dict], cells: list[tuple[str, str]], field: str, condition: str) -> list[float]:
    lookup = defaultdict(list)
    for row in rows:
        value = float(row[field])
        if np.isfinite(value):
            lookup[cell_key(row), condition_key(row)].append(value)
    expected_items = max(map(len, lookup.values()), default=0)
    return [float(np.mean(lookup[cell, condition])) if expected_items and len(lookup[cell, condition]) == expected_items else float("nan")
            for cell in cells]


def model_label(model: str) -> str:
    return model.split("/")[-1].replace("-it", "")


def model_slug(model: str) -> str:
    return model.casefold().replace("/", "__").replace(".", "_")


def ordered_models(rows: list[dict]) -> list[str]:
    found = {row["model"] for row in rows}
    return [model for model in MODEL_ORDER if model in found] + sorted(found - set(MODEL_ORDER))


def draw_panel(
    ax,
    rows: list[dict],
    cells: list[tuple[str, str]],
    field: str,
    title: str,
    legend_title: str,
    show_legend: bool,
) -> None:
    x = np.arange(len(cells))
    width = 0.36
    for offset, condition, color in (
        (-width / 2, "Target", TARGET_COLOR),
        (width / 2, "Control", BASELINE_COLOR),
    ):
        bar_values = values(rows, cells, field, condition)
        ax.bar(
            x + offset,
            bar_values,
            width,
            color=color,
            label=condition,
        )
        for i, value in enumerate(bar_values):
            if not np.isfinite(value):
                ax.text(i + offset, 4, "NA", ha="center", fontsize=7, color="#666666")
    # Missing human control bars must not change the category spacing.
    ax.set_xlim(-0.6, len(cells) - 0.4)
    ax.set_ylim(0, 100)
    ax.set_ylabel("Mean endorsement")
    ax.set_title(title, loc="left", fontsize=11)
    ax.set_xticks(x, [label(cell) for cell in cells])
    ax.tick_params(axis="x", labelsize=8, labelbottom=True, pad=5)
    ax.grid(axis="y", color="#e5e5e5", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    if show_legend:
        ax.legend(
            title=legend_title,
            frameon=False,
            loc="center left",
            bbox_to_anchor=(1.01, 0.5),
        )


def plot_experiment(
    experiment: str, rows: list[dict], model: str, output_dir: Path
) -> None:
    cells, emoji_type, legend_title = experiment_spec(experiment)
    figure_width = max(9.0, len(cells) * 1.4)
    fig, axes = plt.subplots(2, 1, figsize=(figure_width, 6.8), sharex=True)

    for ax, field, title in (
        (axes[0], "human_mean", "Human"),
        (axes[1], "model_score", model_label(model)),
    ):
        draw_panel(ax, rows, cells, field, title, legend_title, ax is axes[0])
    axes[1].set_xlabel("Inference type")
    fig.suptitle(f"{experiment}: {emoji_type}", fontsize=14)
    fig.tight_layout(rect=(0, 0, 0.88, 1))

    slug = experiment.lower().replace(" ", "_")
    for suffix in ("png", "pdf"):
        fig.savefig(
            output_dir / f"{model_slug(model)}_{slug}_human_model_bars.{suffix}",
            dpi=180,
            bbox_inches="tight",
        )
    plt.close(fig)


def plot_all_models(
    experiment: str,
    rows: list[dict],
    output_dir: Path,
    models_per_page: int,
    *, controls: bool = False,
) -> None:
    cells, emoji_type, legend_title = experiment_spec(experiment)
    if controls:
        cells = [(emoji, premise) for emoji, _ in cells for premise in ("Target premise", "Control premise")]
        emoji_type, legend_title = "Post-text premise controls", "Inference"
    models = ordered_models(rows)
    human_rows = [row for row in rows if row["model"] == models[0]]
    slug = experiment.lower().replace(" ", "_")
    if controls:
        slug = "posttext_premise_controls"
    pdf_path = output_dir / f"all_models_{slug}_bar_charts.pdf"

    with PdfPages(pdf_path) as pdf:
        for page, start in enumerate(range(0, len(models), models_per_page), start=1):
            page_models = models[start : start + models_per_page]
            panels = [("Human (control-premise ratings not collected)" if controls else "Human", human_rows, "human_mean")]
            panels.extend(
                (
                    model_label(model),
                    [row for row in rows if row["model"] == model],
                    "model_score",
                )
                for model in page_models
            )
            slots = models_per_page + 1
            width = max(10.5, len(cells) * 1.35)
            fig, axes = plt.subplots(slots, 1, figsize=(width, slots * 2.75), squeeze=False)
            for index, (title, panel_rows, field) in enumerate(panels):
                draw_panel(
                    axes[index, 0],
                    panel_rows,
                    cells,
                    field,
                    title,
                    legend_title,
                    index == 0,
                )
                if controls:
                    if index == 0:
                        axes[index, 0].get_legend().get_texts()[1].set_text("Baseline")
            for index in range(len(panels), slots):
                axes[index, 0].axis("off")
            axes[len(panels) - 1, 0].set_xlabel("Emoji / premise" if controls else "Inference type")
            fig.suptitle(
                f"{experiment}: {emoji_type}, model comparison, page {page}",
                fontsize=13,
            )
            fig.subplots_adjust(left=0.10, right=0.86, top=0.94, bottom=0.07, hspace=0.72)
            pdf.savefig(fig)
            fig.savefig(
                output_dir / f"all_models_{slug}_bar_charts_page{page}.png",
                dpi=180,
                bbox_inches="tight",
            )
            plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = load_rows(args.input)
    for model in sorted({row["model"] for row in rows}):
        model_rows = [row for row in rows if row["model"] == model]
        for experiment in ("Experiment 1", "Experiment 2", "Experiment 3"):
            plot_experiment(
                experiment,
                [row for row in model_rows if row["experiment"] == experiment],
                model,
                args.output_dir,
            )
    for experiment in ("Experiment 1", "Experiment 2", "Experiment 3"):
        plot_all_models(
            experiment,
            [row for row in rows if row["experiment"] == experiment],
            args.output_dir,
            args.models_per_page,
        )
    controls_path = args.input.with_name("posttext_premise_controls.csv")
    if controls_path.exists():
        controls = load_rows(controls_path)
        for row in controls:
            row["experiment"] = "Experiment 1"  # Reuse the existing two-factor panel layout.
            row["environment"] = row["emoji"]
            row["inference_type"] = "Target premise" if row["premise"] == "TargetPremise" else "Control premise"
            row["material_condition"] = "Target" if row["statement_type"] == "target" else "Control"
            for field in ("model_score", "human_mean"):
                row[field] = float(row[field]) if row[field] else float("nan")
        plot_all_models("Experiment 3", controls, args.output_dir, args.models_per_page, controls=True)
        from build_binary_effect_profiles import save_mds
        models = ordered_models(controls)
        features = [(emoji, premise, kind) for emoji, _ in experiment_spec("Experiment 3")[0]
                    for premise in ("TargetPremise", "ControlPremise") for kind in ("target", "baseline")]
        lookup = {(r["model"], r["emoji"], r["premise"], r["statement_type"]): r["model_score"] for r in controls}
        matrix = np.array([[lookup.get((model, *feature), np.nan) for feature in features] for model in models])
        if np.isfinite(matrix).all():
            distances = squareform(pdist(matrix)) / np.sqrt(len(features))
            save_mds(distances, models, args.output_dir.parent / "effect_profiles/posttext_premise_controls_mds",
                     "Post-text: RMS distances of 20 original/control-premise ratings (no human controls)")
    print(f"Wrote emoji comparison charts to {args.output_dir}")


if __name__ == "__main__":
    main()
