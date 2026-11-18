from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram

from build_binary_effect_profiles import save_mds
from plot_item_effects import contrast_tree, corrected
from item_effect_tests import emoji


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EFFECTS = ROOT / "results/emoji/human_tests/emoji_fixed_effect_tests.csv"
DEFAULT_OUTPUT = ROOT / "results/emoji/inference/effect_profiles"
MODEL_ORDER = (
    "Human",
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
CONDITION_ORDER = (
    "Co-text: Each / Existential",
    "Co-text: Each / Universal",
    "Co-text: Each / Interaction",
    "Co-text: Exactly-one / Existential",
    "Co-text: Exactly-one / Universal",
    "Co-text: Exactly-one / Interaction",
    "Co-text: Might / Conditional",
    "Co-text: Negation / Conditional",
    "Co-text: None / Existential",
    "Co-text: None / Universal",
    "Co-text: None / Interaction",
    "Co-text: Unembedded / Manner",
    "Pro-text: All environments",
    "Pro-text: Unembedded",
    "Pro-text: Question",
    "Pro-text: Might",
    "Pro-text: Negation",
    "Pro-text: None",
    "Post-text: All emojis",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot mean item contrasts and exact Wilcoxon tests for emoji effects."
    )
    parser.add_argument("--effects", type=Path, default=DEFAULT_EFFECTS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--alpha", type=float, default=0.05)
    return parser.parse_args()


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def short_model(model: str) -> str:
    return model.split("/")[-1].replace("-Instruct", "").replace("-it", "")


def condition_label(row: dict[str, str]) -> str:
    return row["test"]


def ordered_models(rows: list[dict[str, str]]) -> list[str]:
    found = {row["model"] for row in rows}
    return [model for model in MODEL_ORDER if model in found] + sorted(found - set(MODEL_ORDER))


def matrices(rows: list[dict[str, str]]) -> tuple[list[str], list[str], np.ndarray, np.ndarray]:
    models = ordered_models(rows)
    found_conditions = {condition_label(row) for row in rows}
    conditions = [condition for condition in CONDITION_ORDER if condition in found_conditions]
    conditions.extend(sorted(found_conditions - set(CONDITION_ORDER)))
    lookup = {(row["model"], condition_label(row)): row for row in rows}
    if len(lookup) != len(rows):
        raise ValueError("Duplicate model/test coefficients.")
    missing = [
        (model, condition)
        for model in models
        for condition in conditions
        if (model, condition) not in lookup
    ]
    if missing:
        raise ValueError(f"Incomplete model-condition matrix; first missing cell: {missing[0]}")

    signed = np.array(
        [
            [float(lookup[(model, condition)]["estimate"]) for condition in conditions]
            for model in models
        ]
    )
    detected = np.array([[lookup[(model, condition)]["effect_detected"]
                          for condition in conditions] for model in models], dtype=float)
    return models, conditions, signed, detected


def save_figure(fig: plt.Figure, stem: Path) -> None:
    for suffix in ("png", "pdf"):
        fig.savefig(stem.with_suffix(f".{suffix}"), dpi=200, bbox_inches="tight")
    plt.close(fig)


def clustered_heatmap(
    signed: np.ndarray,
    models: list[str],
    conditions: list[str],
    output_dir: Path,
    *,
    stem: str = "emoji_mean_item_contrasts_clustering",
    title: str = "Emoji effects: mean per-item contrasts and Wilcoxon tests",
    colorbar_label: str = "Mean item contrast (rating points)",
    color_limit: float = 200,
    detected: np.ndarray | None = None,
    alpha: float = .05,
    coverage: str = "",
) -> None:
    # Cluster only on shared observed features; missing values are never imputed.
    tree, distance_matrix, complete = contrast_tree(pd.DataFrame(signed))
    fig, (tree_ax, matrix_ax) = plt.subplots(
        1,
        2,
        figsize=(22, 8),
        gridspec_kw={"width_ratios": (1.2, 5)},
        constrained_layout=True,
    )
    tree_result = dendrogram(
        tree, ax=tree_ax, orientation="left", no_labels=True, color_threshold=None
    )
    tree_ax.set_ylim(10 * len(models), 0)
    order = tree_result["leaves"]
    ordered = signed[order]
    limit = color_limit or max(1.0, float(np.nanmax(abs(ordered))))
    cmap = plt.get_cmap("RdBu_r").copy()
    cmap.set_bad("#d9d9d9")
    image = matrix_ax.imshow(
        ordered, cmap=cmap, aspect="auto", vmin=-limit, vmax=limit
    )
    matrix_ax.set_xticks(
        range(len(conditions)), conditions, rotation=42, ha="right", fontsize=8
    )
    matrix_ax.set_yticks(
        range(len(models)), [short_model(models[index]) for index in order]
    )
    if detected is not None:
        for y, index in enumerate(order):
            for x, value in enumerate(signed[index]):
                label = f"{0 if abs(value) < .5 else value:.0f}" if np.isfinite(value) else "NA"
                label += "*" if detected[index, x] == 1 else ""
                matrix_ax.text(x, y, label, ha="center", va="center", fontsize=7,
                               color="white" if abs(value) > .7 * limit else "black")
    if not complete.all():
        title += f"\nClustering uses {complete.sum()} shared complete features"
    matrix_ax.set_title(title + f"\n* Exact Wilcoxon, BH q < {alpha}; no star does not imply equivalence.\n"
                        + coverage, fontsize=11)
    tree_ax.set_xlabel("RMS item-contrast distance\nAverage linkage", fontsize=9)
    tree_ax.tick_params(axis="y", left=False, labelleft=False)
    fig.colorbar(image, ax=matrix_ax, label=colorbar_label, shrink=0.8)
    save_figure(fig, output_dir / stem)

    write_csv(
        output_dir / f"{stem}_features.csv",
        [{"feature": name, "used_for_clustering": bool(used)} for name, used in zip(conditions, complete)],
    )

    save_mds(distance_matrix, models, output_dir / f"{stem}_mds", "Emoji: RMS mean item-contrast distances")
    write_csv(
        output_dir / f"{stem}_distances.csv",
        [
            {
                "model": model,
                **{
                    other: distance_matrix[row_index, column_index]
                    for column_index, other in enumerate(models)
                },
            }
            for row_index, model in enumerate(models)
        ],
    )
    write_csv(
        output_dir / f"{stem}_order.csv",
        [
            {"position": position, "model": models[index]}
            for position, index in enumerate(order, start=1)
        ],
    )


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if not 0 < args.alpha < 1:
        raise ValueError("--alpha must be between 0 and 1.")
    specifications = pd.read_csv(args.effects, keep_default_na=False, na_values=["NA"])
    tests = emoji(specifications)
    tests = corrected(tests, args.alpha)
    tests.to_csv(args.output_dir / "item_wilcoxon_tests.csv", index=False, na_rep="NA")
    tests.to_csv(ROOT / "results/emoji/human_tests/item_wilcoxon_tests.csv", index=False, na_rep="NA")
    rows = tests.to_dict("records")
    models, conditions, signed, detected = matrices(rows)

    write_csv(
        args.output_dir / "emoji_mean_item_contrast_vectors.csv",
        [
            {"model": model, **dict(zip(conditions, values))}
            for model, values in zip(models, signed)
        ],
    )
    write_csv(
        args.output_dir / "emoji_wilcoxon_detection_vectors.csv",
        [
            {"model": model, **dict(zip(conditions, values))}
            for model, values in zip(models, detected)
        ],
    )
    coverage = []
    for label, human in (("Human", True), ("VLM", False)):
        counts = tests[tests.model.eq("Human") == human].groupby("condition").n_items_expected.max().reindex(
            ["Co-text", "Pro-text", "Post-text"])
        coverage.append(f"{label} items (co/pro/post): " + "/".join(str(int(n)) for n in counts))
    clustered_heatmap(signed, models, conditions, args.output_dir, detected=detected, alpha=args.alpha,
                      coverage="; ".join(coverage))
    print(f"Wrote emoji effect profiles to {args.output_dir}")


if __name__ == "__main__":
    main()
