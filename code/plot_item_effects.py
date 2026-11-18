"""Render item-level effect contrasts and Wilcoxon significance profiles."""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap
from scipy.stats import false_discovery_control

from build_binary_effect_profiles import cluster_profile, save_mds
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import dendrogram, linkage
from build_significant_difference_profiles import ROOT, TEST_ORDER, short_model
import plot_model_barcharts as bars
from item_effect_tests import animation_gesture


def corrected(frame, alpha):
    frame = frame.copy()
    frame["q_value"] = np.nan
    for _, indices in frame.groupby(["model", "input_format"]).groups.items():
        p = frame.loc[indices, "p_value"]
        frame.loc[indices, "q_value"] = false_discovery_control(p.fillna(1), method="bh")
    frame.loc[frame.p_value.isna(), "q_value"] = np.nan
    frame["effect_detected"] = (frame.q_value < alpha).astype(float).where(frame.q_value.notna())
    return frame


def human_diagnostics(path):
    human = pd.read_csv(path, keep_default_na=False)
    interaction = human.condition.isin(["Scalar Implicature", "Supplement"])
    human = human[(interaction & human.tested_effect.eq("Premise x inference")) |
                  (~interaction & human.tested_effect.eq("Inference"))].copy()
    human["model"], human["input_format"] = "Human", "human"
    human["test_design"] = "Lyn original human mixed-effects LRT"
    return human


def contrast_tree(contrasts: pd.DataFrame):
    complete = np.isfinite(contrasts.to_numpy()).all(axis=0)
    if not complete.any():
        raise ValueError("Item-contrast clustering needs at least one shared finite diagnostic.")
    distances = pdist(contrasts.loc[:, complete], metric="euclidean") / np.sqrt(complete.sum())
    return linkage(distances, method="average"), squareform(distances), complete


def regenerate(study: str, alpha: float = 0.05, *, refresh_bars: bool = True):
    base = ROOT / "results" / study
    source = base / "human_tests"
    output = base / "inference/effect_profiles/item_effects"
    output.mkdir(parents=True, exist_ok=True)
    frame = animation_gesture(study)
    vlm = frame[frame.model.ne("Human")]
    if frame.duplicated(["model", "input_format", "test"]).any():
        raise ValueError("Duplicate item-effect diagnostic.")
    if any(set(g.test) != set(TEST_ORDER) for _, g in frame.groupby(["model", "input_format"])):
        raise ValueError("Expected all seven diagnostics per model/format.")
    frame = corrected(frame, alpha)
    frame.to_csv(output / "item_wilcoxon_tests.csv", index=False, na_rep="NA")
    frame.to_csv(source / "item_wilcoxon_tests.csv", index=False, na_rep="NA")
    modes = sorted(vlm.input_format.unique())
    binary_tables, contrast_tables = [], []
    for mode in modes:
        data = frame[frame.input_format.isin([mode, "human"])]
        models = ["Human", *sorted(vlm.model.unique(), key=bars.model_sort_key)]
        contrasts = data.pivot(index="model", columns="test", values="estimate").reindex(index=models, columns=TEST_ORDER)
        binary = data.pivot(index="model", columns="test", values="effect_detected").reindex(index=models, columns=TEST_ORDER)
        binary_tables.append(binary)
        contrast_tables.append(contrasts)
        slug = mode.replace("-", "_")
        contrasts.to_csv(output / f"mean_item_contrasts_{slug}.csv", na_rep="NA")
        tree, distances, complete = contrast_tree(contrasts)
        save_mds(distances, models, output / f"mean_item_contrasts_mds_{slug}",
                 f"{study.title()}: {mode}\nRMS distance of effects by inference type and environment")
        stem = f"mean_item_contrasts_{slug}"
        pd.DataFrame(distances, index=models, columns=models).to_csv(
            output / f"{stem}_distances.csv", index_label="model")
        pd.DataFrame({"feature": contrasts.columns, "used_for_clustering": complete}).to_csv(
            output / f"{stem}_features.csv", index=False)
        binary.astype("Int64").to_csv(output / f"wilcoxon_detection_{slug}.csv", na_rep="NA")
        fig, (tree_ax, ax) = plt.subplots(1, 2, figsize=(15, max(5, .4 * len(models) + 2.8)),
                                         gridspec_kw={"width_ratios": (1.2, 5)}, layout="constrained")
        result = dendrogram(tree, ax=tree_ax, orientation="left", no_labels=True)
        order = result["leaves"]
        tree_ax.set_ylim(10 * len(models), 0)
        tree_ax.set_xlabel("RMS item-contrast distance\nAverage linkage", fontsize=9)
        tree_ax.tick_params(axis="y", left=False, labelleft=False)
        ordered = contrasts.iloc[order]
        ordered_binary = binary.iloc[order]
        pd.DataFrame({"position": range(1, len(models) + 1), "model": ordered.index}).to_csv(
            output / f"{stem}_order.csv", index=False)
        limit = 200  # Common rating-point scale across both studies and every format.
        cmap = plt.get_cmap("RdBu_r").copy()
        cmap.set_bad("#d9d9d9")
        im = ax.imshow(ordered, cmap=cmap, vmin=-limit, vmax=limit, aspect="auto")
        ax.set_xticks(range(7), TEST_ORDER, rotation=35, ha="right", fontsize=9)
        ax.set_yticks(range(len(models)), [short_model(m) for m in ordered.index], fontsize=9)
        fig.suptitle(f"{study.title()} ({mode}): effects by inference type and environment\n"
                     f"Mean item contrasts; {complete.sum()} shared effects; stars: exact Wilcoxon with BH correction", fontsize=11)
        for y, row in enumerate(ordered.to_numpy()):
            for x, value in enumerate(row):
                detected = ordered_binary.iloc[y, x]
                label = (f"{0 if abs(value) < 1e-8 else value:.1f}" + ("*" if detected == 1 else "")) if np.isfinite(value) else "NA"
                ax.text(x, y, label, ha="center", va="center", fontsize=8,
                        color="white" if abs(value) > .7 * limit else "black")
        fig.colorbar(im, ax=ax, label="Mean item contrast (rating points); * BH q < " + str(alpha))
        for ext in ("png", "pdf"):
            fig.savefig(output / f"{stem}.{ext}", dpi=180)
        plt.close(fig)
        if binary.notna().all(axis=0).any():
            cluster_profile(binary.to_numpy(), models, list(TEST_ORDER), output,
                            f"wilcoxon_detection_clustering_{slug}", f"{study.title()} Wilcoxon effect detection: {mode}\nSmall item counts limit power; 0 does not mean no effect",
                            alpha=alpha, effect_presence=True)
        else:
            # A tree cannot be inferred by silently treating unavailable tests as zero.
            fig, ax = plt.subplots(figsize=(12, 7), layout="constrained")
            cmap = ListedColormap(["#eeeeee", "#7BC87C"])
            cmap.set_bad("#d9d9d9")
            ax.imshow(binary, cmap=cmap, vmin=0, vmax=1, aspect="auto")
            ax.set_xticks(range(7), TEST_ORDER, rotation=35, ha="right", fontsize=9)
            ax.set_yticks(range(len(models)), [short_model(m) for m in models], fontsize=9)
            for y, row in enumerate(binary.to_numpy()):
                for x, value in enumerate(row):
                    ax.text(x, y, str(int(value)) if np.isfinite(value) else "NA", ha="center", va="center", fontsize=8)
            ax.set_title(f"{study.title()}: {mode} effect presence (BH q < {alpha})\n"
                         "No tree: no effect has an estimable p-value for every model\n"
                         "1: effect detected; 0: not detected; NA: unavailable", fontsize=10)
            for ext in ("png", "pdf"):
                fig.savefig(output / f"wilcoxon_detection_clustering_{slug}.{ext}", dpi=180)
            plt.close(fig)
    if len(modes) > 1:
        average = sum(binary_tables) / len(binary_tables)
        average.to_csv(output / "wilcoxon_detection_all_modes_mean.csv", na_rep="NA")
        (sum(contrast_tables) / len(contrast_tables)).to_csv(output / "mean_item_contrasts_all_modes.csv", na_rep="NA")
        cluster_profile(average.to_numpy(), models, list(TEST_ORDER), output, "wilcoxon_detection_clustering_all_modes",
                        "Wilcoxon effect detection: mean across formats (not a new significance test)",
                        binary=False, alpha=alpha, effect_presence=True)
    if refresh_bars:
        regenerate_bars(base, source / "llm_fixed_effect_cells.csv")
    print(f"Wrote item-level Wilcoxon profiles to {output}")


def regenerate_bars(base: Path, cells_path: Path):
    cells = pd.read_csv(cells_path, keep_default_na=False)
    cells["human_mean"] = pd.to_numeric(cells.human_mean, errors="coerce")
    # Prefix matching also covers story 4 in gestures; do not silently drop it.
    bars.INFERENCE_ORDER = [(label, (label.replace(" ", ""),)) for label in
                            ("Scalar implicature", "Presupposition", "Supplement", "Homogeneity")]
    bars.INFERENCE_ORDER[0] = ("Scalar implicature", ("ScalarImplicature",))
    keys = ["model", "input_format", "source_video", "condition", "environment", "Story", "GesturalPremise"]
    def paired_rows(data, value):
        table = data.pivot(index=keys, columns="statement_type", values=value).reset_index()
        table = table.rename(columns={"model": "raw_model", "input_format": "input_mode",
                                      "target": "target_score", "baseline": "baseline_score",
                                      "environment": "condition_label"})
        table["task_family"] = table.condition.str.replace(" ", "") + table.Story.astype(str)
        table["premise"] = table.GesturalPremise + "Premise"
        table["condition"] = table.condition_label
        table["model"] = table.raw_model + table.input_mode.map(lambda mode: "" if mode == "split" else f" [{mode}]")
        table["target_minus_baseline"] = table.target_score - table.baseline_score
        return table.to_dict("records")
    output = base / "inference/bar_charts"
    rows = paired_rows(cells, "value")
    human = cells.dropna(subset=["human_mean"]).drop_duplicates(["source_video", "statement_type"]).copy()
    human["model"], human["input_format"] = "Human", "human"
    human_rows = paired_rows(human, "human_mean")
    for mode in cells.input_format.unique():
        mode_rows = [r for r in rows if r["input_mode"] == mode]
        bars.plot_combined_input_mode(mode, mode_rows, human_rows, output, ("png", "pdf"), 3)
        for model in sorted({r["raw_model"] for r in mode_rows}, key=bars.model_sort_key):
            label = model if mode == "split" else f"{model}_{mode}"
            bars.plot_model(label, [r for r in mode_rows if r["raw_model"] == model],
                            output, ("png", "pdf"), experiment=base.name.title())
    if base.name == "gesture":
        (base / "inference/comparison").mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(base / "inference/comparison/ratings_paired.csv", index=False)
