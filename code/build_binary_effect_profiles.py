from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from statistics import mean

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist, squareform


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EFFECTS = ROOT / "results/model_comparison/model_human_effects.csv"
DEFAULT_HUMAN = ROOT / "materials/stats/TSC_EXP_2_human_effects.csv"
DEFAULT_OUTPUT = ROOT / "results/binary_effect_profiles"
TESTS = ("Scalar implicature", "Homogeneity", "Presupposition", "Supplement")
CONDITION_TESTS = (
    "Scalar implicature / Positive",
    "Scalar implicature / Negative",
    "Homogeneity / Positive",
    "Homogeneity / Negative",
    "Presupposition / Question",
    "Presupposition / None",
    "Supplement / Unembedded",
)
MODES = ("pure-text", "pure-video", "split")
MODEL_ORDER = (
    "Qwen/Qwen2.5-VL-3B-Instruct",
    "Qwen/Qwen2.5-VL-7B-Instruct",
    "Qwen/Qwen2.5-VL-32B-Instruct",
    "Qwen/Qwen3-VL-2B-Instruct",
    "Qwen/Qwen3-VL-8B-Instruct",
    "Qwen/Qwen3.8-27B",
    "google/gemma-4-12B-it",
)


def ordered_models(rows: list[dict]) -> list[str]:
    models = {row["model"] for row in rows}
    known = [model for model in MODEL_ORDER if model in models]
    return known + sorted(models.difference(known))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def key(condition: str, environment: str) -> tuple[str, str]:
    return condition.casefold(), environment.casefold()


def standardized_discrepancies(
    effect_rows: list[dict[str, str]], human_rows: list[dict[str, str]], scale_field: str
) -> list[dict]:
    human = {key(row["Condition"], row["Environment"]): row for row in human_rows}
    output = []
    for row in effect_rows:
        human_row = human[key(row["human_condition"], row["human_environment"])]
        model_effect = float(row["model_effect"])
        human_effect = float(human_row["HUMAN_EFFECT"])
        scale = float(human_row[scale_field])
        discrepancy = model_effect - human_effect
        output.append(
            {
                "model": row["raw_model"],
                "input_mode": row["input_mode"],
                "inference_type": row["inference_group"],
                "condition": row["condition"],
                "model_effect": model_effect,
                "human_effect": human_effect,
                "human_scale": scale,
                "signed_discrepancy": discrepancy,
                "absolute_discrepancy": abs(discrepancy),
                "standardized_discrepancy": discrepancy / scale,
                "absolute_standardized_discrepancy": abs(discrepancy) / scale,
            }
        )
    return output


def merge_inference_types(rows: list[dict]) -> list[dict]:
    grouped: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for row in rows:
        grouped[(row["model"], row["input_mode"], row["inference_type"])].append(row)

    output = []
    for (model, mode, inference_type), condition_rows in grouped.items():
        output.append(
            {
                "model": model,
                "input_mode": mode,
                "inference_type": inference_type,
                "n_conditions": len(condition_rows),
                "mean_absolute_discrepancy": mean(
                    row["absolute_discrepancy"] for row in condition_rows
                ),
                "mean_signed_z": mean(
                    row["standardized_discrepancy"] for row in condition_rows
                ),
                "mean_absolute_z": mean(
                    row["absolute_standardized_discrepancy"] for row in condition_rows
                ),
            }
        )
    return output


def binary_profiles(rows: list[dict], threshold: float) -> list[dict]:
    lookup = {
        (row["model"], row["input_mode"], row["inference_type"]): row["mean_absolute_z"]
        for row in rows
    }
    models = ordered_models(rows)
    output = []
    for mode in MODES:
        for model in models:
            values = [int(lookup[(model, mode, test)] < threshold) for test in TESTS]
            output.append(
                {
                    "model": model,
                    "input_mode": mode,
                    "threshold": threshold,
                    **{test: value for test, value in zip(TESTS, values)},
                    "profile": "".join(str(value) for value in values),
                    "n_human_like": sum(values),
                }
            )
    return output


def continuous_profiles(rows: list[dict]) -> list[dict]:
    lookup = {
        (row["model"], row["input_mode"], row["inference_type"]): row["mean_absolute_z"]
        for row in rows
    }
    models = ordered_models(rows)
    output = []
    for mode in MODES:
        for model in models:
            values = [lookup[(model, mode, test)] for test in TESTS]
            output.append(
                {
                    "model": model,
                    "input_mode": mode,
                    **{test: value for test, value in zip(TESTS, values)},
                    "mean_absolute_z": mean(values),
                }
            )
    return output


def condition_profiles(rows: list[dict], threshold: float | None = None) -> list[dict]:
    lookup = {
        (row["model"], row["input_mode"], f'{row["inference_type"]} / {row["condition"]}'):
        row["absolute_standardized_discrepancy"]
        for row in rows
    }
    output = []
    for mode in MODES:
        for model in ordered_models(rows):
            values = [lookup[(model, mode, test)] for test in CONDITION_TESTS]
            if threshold is None:
                output.append(
                    {
                        "model": model,
                        "input_mode": mode,
                        **{test: value for test, value in zip(CONDITION_TESTS, values)},
                        "mean_absolute_z": mean(values),
                    }
                )
            else:
                binary = [int(value < threshold) for value in values]
                output.append(
                    {
                        "model": model,
                        "input_mode": mode,
                        "threshold": threshold,
                        **{test: value for test, value in zip(CONDITION_TESTS, binary)},
                        "profile": "".join(str(value) for value in binary),
                        "n_human_like": sum(binary),
                    }
                )
    return output


def short_name(model: str) -> str:
    return model.split("/")[-1].replace("-Instruct", "").replace("-it", "")


def plot_profiles(
    rows: list[dict],
    threshold: float,
    output_dir: Path,
    tests: tuple[str, ...] = TESTS,
    output_name: str = "binary_effect_profiles",
    title_label: str = "effect profiles",
) -> None:
    models = ordered_models(rows)
    lookup = {(row["model"], row["input_mode"]): row for row in rows}
    width = 12.5 if len(tests) <= 4 else 17
    fig, axes = plt.subplots(1, 3, figsize=(width, 6), sharey=True, constrained_layout=True)
    cmap = ListedColormap(["#d9d9d9", "#62c98d"])

    for ax, mode in zip(axes, MODES):
        matrix = np.array([[lookup[(model, mode)][test] for test in tests] for model in models])
        ax.imshow(matrix, cmap=cmap, vmin=0, vmax=1, aspect="auto")
        ax.set_xticks(range(len(tests)), tests, rotation=35, ha="right")
        ax.set_yticks(range(len(models)), [short_name(model) for model in models])
        ax.set_title(mode)
        for y in range(len(models)):
            for x in range(len(tests)):
                ax.text(x, y, str(matrix[y, x]), ha="center", va="center", fontsize=10)

    fig.suptitle(
        f"Binary {title_label}: absolute standardized discrepancy < {threshold:g} human SD"
    )
    for extension in ("png", "pdf"):
        fig.savefig(output_dir / f"{output_name}.{extension}", dpi=200)
    plt.close(fig)


def plot_continuous_profiles(
    rows: list[dict],
    output_dir: Path,
    tests: tuple[str, ...] = TESTS,
    output_name: str = "continuous_effect_profiles",
    title_label: str = "effect discrepancy profiles",
    colorbar_label: str = "Mean absolute standardized effect discrepancy (human SD units)",
) -> None:
    models = ordered_models(rows)
    lookup = {(row["model"], row["input_mode"]): row for row in rows}
    values = [row[test] for row in rows for test in tests]
    width = 12.5 if len(tests) <= 4 else 17
    fig, axes = plt.subplots(1, 3, figsize=(width, 6), sharey=True, constrained_layout=True)

    for ax, mode in zip(axes, MODES):
        matrix = np.array([[lookup[(model, mode)][test] for test in tests] for model in models])
        image = ax.imshow(
            matrix,
            cmap="RdYlGn_r",
            vmin=0,
            vmax=max(values),
            aspect="auto",
        )
        ax.set_xticks(range(len(tests)), tests, rotation=35, ha="right")
        ax.set_yticks(range(len(models)), [short_name(model) for model in models])
        ax.set_title(mode)
        for y in range(len(models)):
            for x in range(len(tests)):
                ax.text(x, y, f"{matrix[y, x]:.2f}", ha="center", va="center", fontsize=9)

    fig.colorbar(
        image,
        ax=axes,
        label=colorbar_label,
        shrink=0.85,
    )
    fig.suptitle(f"Continuous model–human {title_label} (lower is better)")
    for extension in ("png", "pdf"):
        fig.savefig(output_dir / f"{output_name}.{extension}", dpi=200)
    plt.close(fig)


def profile_matrix(
    rows: list[dict], modes: tuple[str, ...], tests: tuple[str, ...] = TESTS
) -> tuple[list[str], np.ndarray]:
    models = ordered_models(rows)
    lookup = {(row["model"], row["input_mode"]): row for row in rows}
    matrix = np.array(
        [
            [lookup[(model, mode)][test] for mode in modes for test in tests]
            for model in models
        ],
        dtype=float,
    )
    return models, matrix


def plot_clustering(
    rows: list[dict],
    modes: tuple[str, ...],
    metric: str,
    method: str,
    title: str,
    output_stem: Path,
    tests: tuple[str, ...] = TESTS,
) -> None:
    models, matrix = profile_matrix(rows, modes, tests)
    distances = pdist(matrix, metric=metric)
    tree = linkage(distances, method=method)

    fig, (tree_ax, matrix_ax) = plt.subplots(
        2, 1, figsize=(9, 9), gridspec_kw={"height_ratios": (1, 3)}, constrained_layout=True
    )
    tree_result = dendrogram(
        tree,
        ax=tree_ax,
        labels=[short_name(model) for model in models],
        leaf_rotation=30,
        leaf_font_size=9,
        color_threshold=None,
    )
    tree_ax.set_ylabel("Linkage distance")
    tree_ax.set_title(title)

    order = tree_result["leaves"]
    cluster_order = [models[index] for index in order]
    distance_matrix = squareform(distances)
    image = matrix_ax.imshow(distance_matrix, cmap="YlOrRd", vmin=0)
    labels = [short_name(model) for model in models]
    matrix_ax.set_xticks(range(len(labels)), labels, rotation=35, ha="right")
    matrix_ax.set_yticks(range(len(labels)), labels)
    matrix_ax.set_title(f"Pairwise {metric} distance")
    for y in range(len(labels)):
        for x in range(len(labels)):
            matrix_ax.text(x, y, f"{distance_matrix[y, x]:.2f}", ha="center", va="center")
    fig.colorbar(image, ax=matrix_ax, shrink=0.8, label="Distance")

    write_csv(
        output_stem.with_name(f"{output_stem.name}_distances.csv"),
        [
            {
                "model": model,
                **{
                    other_model: distance_matrix[row_index, column_index]
                    for column_index, other_model in enumerate(models)
                },
            }
            for row_index, model in enumerate(models)
        ],
    )
    write_csv(
        output_stem.with_name(f"{output_stem.name}_leaf_order.csv"),
        [{"position": index + 1, "model": model} for index, model in enumerate(cluster_order)],
    )
    for extension in ("png", "pdf"):
        fig.savefig(output_stem.with_suffix(f".{extension}"), dpi=200)
    plt.close(fig)


def plot_cluster_analyses(
    continuous: list[dict], sensitivity: list[dict], thresholds: tuple[float, ...], output_dir: Path
) -> None:
    cluster_dir = output_dir / "clustering"
    cluster_dir.mkdir(parents=True, exist_ok=True)

    for modes in ((mode,) for mode in MODES):
        mode = modes[0]
        plot_clustering(
            continuous,
            modes,
            metric="euclidean",
            method="ward",
            title=f"Continuous effect-profile clustering: {mode}",
            output_stem=cluster_dir / f"continuous_{mode.replace('-', '_')}_clustering",
        )
    plot_clustering(
        continuous,
        MODES,
        metric="euclidean",
        method="ward",
        title="Continuous effect-profile clustering: all input formats",
        output_stem=cluster_dir / "continuous_all_modes_clustering",
    )

    for threshold in thresholds:
        threshold_rows = [row for row in sensitivity if row["threshold"] == threshold]
        threshold_label = f"{threshold:g}".replace(".", "p")
        plot_clustering(
            threshold_rows,
            MODES,
            metric="hamming",
            method="average",
            title=f"Binary effect-profile clustering: all formats, threshold {threshold:g}",
            output_stem=cluster_dir / f"binary_tau_{threshold_label}_all_modes_clustering",
        )


def plot_condition_cluster_analyses(
    continuous: list[dict], sensitivity: list[dict], thresholds: tuple[float, ...], output_dir: Path
) -> None:
    cluster_dir = output_dir / "condition_clustering"
    cluster_dir.mkdir(parents=True, exist_ok=True)

    for mode in MODES:
        plot_clustering(
            continuous,
            (mode,),
            metric="euclidean",
            method="ward",
            title=f"Continuous condition-level clustering: {mode}",
            output_stem=cluster_dir / f"continuous_{mode.replace('-', '_')}_clustering",
            tests=CONDITION_TESTS,
        )
    plot_clustering(
        continuous,
        MODES,
        metric="euclidean",
        method="ward",
        title="Continuous condition-level clustering: all input formats",
        output_stem=cluster_dir / "continuous_all_modes_clustering",
        tests=CONDITION_TESTS,
    )

    for threshold in thresholds:
        threshold_rows = [row for row in sensitivity if row["threshold"] == threshold]
        threshold_label = f"{threshold:g}".replace(".", "p")
        plot_clustering(
            threshold_rows,
            MODES,
            metric="hamming",
            method="average",
            title=f"Binary condition-level clustering: all formats, threshold {threshold:g}",
            output_stem=cluster_dir / f"binary_tau_{threshold_label}_all_modes_clustering",
            tests=CONDITION_TESTS,
        )

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build binary model profiles from standardized model-human effect discrepancies."
    )
    parser.add_argument("--effects", type=Path, default=DEFAULT_EFFECTS)
    parser.add_argument("--human-effects", type=Path, default=DEFAULT_HUMAN)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--threshold", type=float, default=1.0)
    parser.add_argument("--sensitivity-thresholds", type=float, nargs="+", default=(0.5, 1.0, 1.5))
    parser.add_argument("--scale", choices=("sd", "se"), default="sd")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.threshold <= 0 or any(value <= 0 for value in args.sensitivity_thresholds):
        raise ValueError("Thresholds must be positive.")

    scale_field = "HUMAN_SD" if args.scale == "sd" else "HUMAN_SE"
    condition_rows = standardized_discrepancies(
        read_csv(args.effects), read_csv(args.human_effects), scale_field
    )
    merged_rows = merge_inference_types(condition_rows)
    continuous = continuous_profiles(merged_rows)
    profiles = binary_profiles(merged_rows, args.threshold)
    sensitivity = [
        row
        for threshold in args.sensitivity_thresholds
        for row in binary_profiles(merged_rows, threshold)
    ]
    condition_continuous = condition_profiles(condition_rows)
    condition_binary = condition_profiles(condition_rows, args.threshold)
    condition_sensitivity = [
        row
        for threshold in args.sensitivity_thresholds
        for row in condition_profiles(condition_rows, threshold)
    ]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "condition_standardized_discrepancies.csv", condition_rows)
    write_csv(args.output_dir / "merged_inference_discrepancies.csv", merged_rows)
    write_csv(args.output_dir / "continuous_effect_profiles.csv", continuous)
    write_csv(args.output_dir / "binary_effect_profiles.csv", profiles)
    write_csv(args.output_dir / "binary_effect_profile_sensitivity.csv", sensitivity)
    write_csv(args.output_dir / "continuous_condition_profiles.csv", condition_continuous)
    write_csv(args.output_dir / "binary_condition_profiles.csv", condition_binary)
    write_csv(
        args.output_dir / "binary_condition_profile_sensitivity.csv", condition_sensitivity
    )
    for mode in MODES:
        write_csv(
            args.output_dir / f"continuous_effect_profiles_{mode.replace('-', '_')}.csv",
            [row for row in continuous if row["input_mode"] == mode],
        )
        write_csv(
            args.output_dir / f"binary_effect_profiles_{mode.replace('-', '_')}.csv",
            [row for row in profiles if row["input_mode"] == mode],
        )
        write_csv(
            args.output_dir / f"continuous_condition_profiles_{mode.replace('-', '_')}.csv",
            [row for row in condition_continuous if row["input_mode"] == mode],
        )
        write_csv(
            args.output_dir / f"binary_condition_profiles_{mode.replace('-', '_')}.csv",
            [row for row in condition_binary if row["input_mode"] == mode],
        )
    plot_profiles(profiles, args.threshold, args.output_dir)
    plot_continuous_profiles(continuous, args.output_dir)
    plot_profiles(
        condition_binary,
        args.threshold,
        args.output_dir,
        tests=CONDITION_TESTS,
        output_name="binary_condition_profiles",
        title_label="condition-level profiles",
    )
    plot_continuous_profiles(
        condition_continuous,
        args.output_dir,
        tests=CONDITION_TESTS,
        output_name="continuous_condition_profiles",
        title_label="condition-level discrepancy profiles",
        colorbar_label="Absolute standardized effect discrepancy (human SD units)",
    )
    plot_cluster_analyses(
        continuous, sensitivity, tuple(args.sensitivity_thresholds), args.output_dir
    )
    plot_condition_cluster_analyses(
        condition_continuous,
        condition_sensitivity,
        tuple(args.sensitivity_thresholds),
        args.output_dir,
    )
    print(f"Wrote binary effect profiles to {args.output_dir}")


if __name__ == "__main__":
    main()
