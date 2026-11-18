from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap
from scipy.cluster.hierarchy import cophenet, dendrogram, linkage
from scipy.spatial.distance import pdist, squareform

from build_significant_difference_profiles import (
    MODEL_ORDER, MODES, TEST_ORDER, adjust_p_values, main, profile_matrix, short_model,
)


def save_matrix(matrix: np.ndarray, models: list[str], columns: list[str], path: Path) -> None:
    pd.DataFrame(matrix, index=models, columns=columns).to_csv(
        path, index_label="model", na_rep="NA"
    )


def save_mds(distances: np.ndarray, models: list[str], path: Path, title: str) -> None:
    """Metric MDS of explicit distances; identical profiles share a label."""
    from sklearn.manifold import MDS

    if len(models) < 3:
        return
    if (not np.isfinite(distances).all() or not np.allclose(distances, distances.T)
            or (distances < 0).any() or not np.allclose(np.diag(distances), 0)):
        raise ValueError("MDS requires finite symmetric distances.")
    if np.max(distances) == 0:
        coordinates, stress = np.zeros((len(models), 2)), 0.0
    else:
        fit = MDS(n_components=2, dissimilarity="precomputed", random_state=42,
                  n_init=8, max_iter=1000, normalized_stress=True)
        coordinates = fit.fit_transform(distances)
        stress = float(fit.stress_)
    pd.DataFrame({"model": models, "dimension_1": coordinates[:, 0],
                  "dimension_2": coordinates[:, 1], "normalized_stress": stress,
                  "seed": 42}).to_csv(path.with_suffix(".csv"), index=False)
    plot_mds(coordinates, distances, models, path, title, stress)


def plot_mds(coordinates, distances, models, path, title, stress):
    """Render MDS coordinates separately so styling never requires refitting."""
    from adjustText import adjust_text

    fig = plt.figure(figsize=(18, 13))
    ax = fig.add_axes((.12, .10, .76, .72))
    labeled = set()
    labels, targets = [], []
    for i, model in enumerate(models):
        if i in labeled:
            continue
        group = [j for j in range(len(models)) if np.allclose(distances[i], distances[j])]
        labeled.update(group)
        x, y = coordinates[i]
        ax.scatter(x, y, color="#222222" if "Human" in [models[j] for j in group] else "#397aab", s=55)
        labels.append(ax.text(x, y, "\n".join(short_model(models[j]) for j in group), fontsize=20))
        targets.append((x, y))
    ax.set_title(f"{title}\nMetric MDS; normalized stress = {stress:.3f}; seed = 42", fontsize=20)
    ax.set_xlabel("MDS-1", fontsize=24)
    ax.set_ylabel("MDS-2", fontsize=24)
    ax.set_xticks([])
    ax.set_yticks([])
    # Fixed frame geometry, with equal physical units in both dimensions.
    aspect = (.76 * 18) / (.72 * 13)
    center = (coordinates.min(axis=0) + coordinates.max(axis=0)) / 2
    height = max(np.ptp(coordinates[:, 1]), np.ptp(coordinates[:, 0]) / aspect, 1) * 1.3
    ax.set_xlim(center[0] - height * aspect / 2, center[0] + height * aspect / 2)
    ax.set_ylim(center[1] - height / 2, center[1] + height / 2)
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(False)
    target_x, target_y = np.asarray(targets).T
    adjust_text(labels, x=target_x, y=target_y, target_x=target_x, target_y=target_y,
                ax=ax, iter_lim=2000, expand=(1.12, 1.3),
                force_text=(.8, 1), force_pull=(.01, .01), prevent_crossings=False)
    for ext in ("png", "pdf"):
        fig.savefig(path.with_suffix("." + ext), dpi=180)
    plt.close(fig)


def cluster_profile(
    matrix: np.ndarray,
    models: list[str],
    columns: list[str],
    output_dir: Path,
    stem: str,
    title: str,
    *,
    binary: bool = True,
    alpha: float = 0.05,
    effect_presence: bool = False,
) -> None:
    # Every pair is compared on the same observed features; NA never becomes zero.
    complete = np.isfinite(matrix).all(axis=0)
    if not complete.any():
        raise ValueError("No shared observed diagnostics remain for clustering.")
    observed = matrix[:, complete]
    distances = pdist(observed, metric="hamming" if binary else "cityblock")
    if not binary:
        distances /= observed.shape[1]
    tree = linkage(distances, method="average")
    distance_matrix = squareform(distances)
    save_mds(distance_matrix, models, output_dir / f"{stem}_mds", title)
    save_matrix(distance_matrix, models, models, output_dir / f"{stem}_distances.csv")
    save_matrix(1 - distance_matrix, models, models, output_dir / f"{stem}_similarities.csv")
    pd.DataFrame({"feature": columns, "used_for_clustering": complete}).to_csv(
        output_dir / f"{stem}_features.csv", index=False
    )

    fig, (tree_ax, ax) = plt.subplots(
        1, 2, figsize=(20 if len(columns) > 7 else 14, 7),
        gridspec_kw={"width_ratios": (1.2, 5)}, constrained_layout=True
    )
    if np.any(distances):
        result = dendrogram(tree, ax=tree_ax, orientation="left", no_labels=True)
        tree_ax.invert_yaxis()
        order = result["leaves"]
    else:
        order = list(range(len(models)))
        tree_ax.axis("off")
        tree_ax.text(.5, .5, "No resolved clustering\nAll distances are zero", ha="center", va="center", transform=tree_ax.transAxes)
    ordered = matrix[order]
    cmap = ListedColormap(["#eeeeee", "#7BC87C"] if effect_presence else ["#7BC87C", "#D65F5F"]) if binary else plt.get_cmap("Greens" if effect_presence else "RdYlGn_r").copy()
    cmap.set_bad("#d9d9d9")
    image = ax.imshow(ordered, cmap=cmap, vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(columns)), columns, rotation=43, ha="right", fontsize=8)
    ax.set_yticks(range(len(models)), [short_model(models[i]) for i in order])
    ax.set_title(
        f"{title} (q < {alpha:g})\nAverage linkage; {complete.sum()} shared features; gray = unavailable"
        + ("\nExploratory: effect detection, not human-LLM difference tests" if effect_presence else
           "\nExploratory: singular fits retained; 0 does not establish equivalence"),
        fontsize=10
    )
    tree_ax.set_xlabel("Hamming distance" if binary else "Mean absolute\nproportion difference")
    tree_ax.tick_params(axis="y", left=False, labelleft=False)
    if binary:
        colorbar = fig.colorbar(image, ax=ax, ticks=[0.25, 0.75], shrink=0.7)
        colorbar.ax.set_yticklabels(["0: no detected effect", "1: detected effect"] if effect_presence else
                                   ["0: no detected difference", "1: detected difference"])
    else:
        fig.colorbar(image, ax=ax, label=f"Proportion of three formats with q < {alpha:g}", shrink=0.7)
    for y, values in enumerate(ordered):
        for x, value in enumerate(values):
            label = ("{:.0f}" if binary else "{:.2f}").format(value) if np.isfinite(value) else "NA"
            ax.text(x, y, label, ha="center", va="center", fontsize=8)
    for suffix in ("png", "pdf"):
        fig.savefig(output_dir / f"{stem}.{suffix}", dpi=200)
    plt.close(fig)
    pd.DataFrame({"position": range(1, len(models) + 1), "model": [models[i] for i in order]}).to_csv(
        output_dir / f"{stem}_order.csv", index=False
    )


def render_profiles(frame: pd.DataFrame, output_dir: Path) -> None:
    rows = frame.to_dict("records")
    alpha = float(frame["alpha"].iloc[0])
    matrices = []
    for mode in MODES:
        models, tests, matrix = profile_matrix(rows, mode)
        matrices.append(matrix)
        slug = mode.replace("-", "_")
        pd.DataFrame(matrix, index=models, columns=tests).astype("Int64").to_csv(
            output_dir / f"binary_condition_profiles_{slug}.csv", index_label="model", na_rep="NA"
        )
        cluster_profile(
            matrix, models, tests, output_dir, f"binary_{slug}_clustering",
            f"BH-adjusted human-LLM differences: {mode}", alpha=alpha
        )

    # Concatenated binary decisions and averaged decisions answer different questions.
    columns = [f"{mode}: {test}" for mode in MODES for test in tests]
    all_formats = np.concatenate(matrices, axis=1)
    pd.DataFrame(all_formats, index=models, columns=columns).astype("Int64").to_csv(
        output_dir / "binary_all_modes_profiles.csv", index_label="model", na_rep="NA"
    )
    cluster_profile(
        all_formats, models, columns, output_dir, "significant_difference_clustering",
        "Binary differences across all formats (concatenated decisions)", alpha=alpha
    )
    proportions = np.mean(matrices, axis=0)
    save_matrix(proportions, models, tests, output_dir / "aggregate_significant_difference_profiles.csv")
    cluster_profile(
        proportions, models, tests, output_dir, "aggregate_significant_difference_clustering",
        "Mean of binary decisions across all three formats (not a binary vector)", binary=False, alpha=alpha
    )


SENSITIVITY_VARIANTS = ("lyn_final", "lyn_with_story", "uncorrelated_slopes", "intercepts_only")


def clustering_sensitivity(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for mode in MODES:
        matrices = {
            variant: frame[(frame["variant"] == variant) & (frame["input_mode"] == mode)]
            .pivot(index="model", columns="test", values="significantly_different")
            .reindex(index=MODEL_ORDER, columns=TEST_ORDER).to_numpy()
            for variant in SENSITIVITY_VARIANTS
        }
        complete = np.isfinite(np.stack(list(matrices.values()))).all(axis=(0, 1))
        reference = None
        for variant, matrix in matrices.items():
            row = {"variant": variant, "input_mode": mode, "n_shared_features": int(complete.sum()),
                   "mean_distance_change": np.nan, "max_distance_change": np.nan,
                   "mean_cophenetic_change": np.nan, "n_unique_profiles": np.nan}
            if complete.any():
                observed = matrix[:, complete]
                distances = pdist(observed, metric="hamming")
                cophenetic = cophenet(linkage(distances, method="average"))
                if reference is None:
                    reference = (distances, cophenetic)
                row.update(
                    mean_distance_change=float(np.mean(abs(distances - reference[0]))),
                    max_distance_change=float(np.max(abs(distances - reference[0]))),
                    mean_cophenetic_change=float(np.mean(abs(cophenetic - reference[1]))),
                    n_unique_profiles=len(np.unique(observed, axis=0)),
                )
            rows.append(row)
    return pd.DataFrame(rows)


def sensitivity_panel(ax, matrix: pd.DataFrame, title: str, *, maximum=None, good_high=False) -> None:
    cmap = plt.get_cmap("YlGn" if good_high else "YlOrRd").copy()
    cmap.set_bad("#d9d9d9")
    image = ax.imshow(matrix.to_numpy(), cmap=cmap, aspect="auto", vmin=0, vmax=maximum)
    labels = ["Lyn final", "+ Story", "Uncorrelated slopes", "Intercepts only"]
    ax.set_xticks(range(len(matrix.columns)), labels, rotation=20, ha="right", fontsize=8)
    ax.set_yticks(range(len(matrix.index)), matrix.index, fontsize=8)
    ax.set_title(title, fontsize=11)
    for y, values in enumerate(matrix.to_numpy()):
        for x, value in enumerate(values):
            ax.text(x, y, f"{value:.2f}" if np.isfinite(value) else "NA", ha="center", va="center", fontsize=8)
    ax.figure.colorbar(image, ax=ax, shrink=0.75)


def render_sensitivity(folder: Path, alpha: float = 0.05) -> None:
    """Compare predefined structures, without selecting or promoting a main model."""
    frame = pd.read_csv(folder / "random_effects_tests.csv", keep_default_na=False).rename(
        columns={"input_format": "input_mode"}
    )
    numbers = ["p_value", "aic_full", "residual_rmse", "residual_outlier_fraction", "absolute_residual_fitted_rho"]
    for column in numbers:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    for column in ("full_converged", "reduced_converged", "singular_fit"):
        frame[column] = frame[column].astype(str).str.lower().eq("true")
    frame["fitted"] = frame["analysis_status"].eq("ok")
    frame["converged"] = frame["fitted"] & frame["full_converged"] & frame["reduced_converged"]
    frame["regular_fit"] = frame["converged"] & ~frame["singular_fit"]
    frame.loc[~frame["converged"], "p_value"] = np.nan
    frame = pd.concat([adjust_p_values(group.copy(), alpha) for _, group in frame.groupby("variant", sort=False)], ignore_index=True)
    frame["test"] = frame["condition"].replace({"Scalar Implicature": "Scalar implicature"}) + ": " + frame["environment"]
    keys = ["model", "input_mode", "condition", "environment"]
    if frame.duplicated([*keys, "variant"]).any():
        raise ValueError("Duplicate sensitivity cases.")
    reference = frame[frame["variant"] == "lyn_final"][[*keys, "significantly_different"]].rename(
        columns={"significantly_different": "baseline_decision"}
    )
    frame = frame.merge(reference, on=keys, validate="many_to_one")
    comparable = frame["significantly_different"].notna() & frame["baseline_decision"].notna()
    frame["decision_changed"] = frame["significantly_different"].ne(frame["baseline_decision"]).astype(float).where(comparable)
    eligible_aic = frame["aic_full"].where(frame["fitted"] & frame["full_converged"])
    minimum = frame.assign(eligible_aic=eligible_aic).groupby(keys)["eligible_aic"].transform("min")
    frame["delta_aic"] = (eligible_aic - minimum).where(eligible_aic.notna())
    frame.to_csv(folder / "random_effects_decisions.csv", index=False, na_rep="NA")
    summary = frame.groupby(["variant", "test"], as_index=False).agg(
        n_cases=("fitted", "size"), n_fitted=("fitted", "sum"), n_converged=("converged", "sum"),
        n_singular=("singular_fit", "sum"), n_regular=("regular_fit", "sum"),
        regular_fraction=("regular_fit", "mean"), median_delta_aic=("delta_aic", "median"),
        n_comparable_decisions=("decision_changed", "count"), n_changed=("decision_changed", "sum"),
        decision_change_fraction=("decision_changed", "mean"),
        median_residual_rmse=("residual_rmse", "median"),
        median_outlier_fraction=("residual_outlier_fraction", "median"),
        median_absolute_residual_fitted_rho=("absolute_residual_fitted_rho", lambda values: values.abs().median()),
    )
    order = pd.MultiIndex.from_product([TEST_ORDER, SENSITIVITY_VARIANTS], names=["test", "variant"])
    summary = summary.set_index(["test", "variant"]).reindex(order).reset_index()
    summary.to_csv(folder / "random_effects_summary.csv", index=False, na_rep="NA")
    clustering = clustering_sensitivity(frame)
    clustering.to_csv(folder / "clustering_stability.csv", index=False, na_rep="NA")

    def diagnostic_matrix(column):
        return summary.pivot(index="test", columns="variant", values=column).reindex(index=TEST_ORDER, columns=SENSITIVITY_VARIANTS)

    fig, axes = plt.subplots(2, 2, figsize=(15, 10), constrained_layout=True)
    sensitivity_panel(axes[0, 0], diagnostic_matrix("regular_fraction"), "Converged, non-singular fit fraction", maximum=1, good_high=True)
    sensitivity_panel(axes[0, 1], diagnostic_matrix("median_delta_aic"), "Median full-model delta AIC (lower is better)")
    sensitivity_panel(axes[1, 0], diagnostic_matrix("decision_change_fraction"), "BH decision changes versus Lyn final", maximum=1)
    matrix = clustering.pivot(index="input_mode", columns="variant", values="mean_distance_change").reindex(index=MODES, columns=SENSITIVITY_VARIANTS)
    sensitivity_panel(axes[1, 1], matrix, "Mean Hamming-distance change versus Lyn final", maximum=1)
    fig.suptitle("Random-effects sensitivity: no automatic model selection\nAIC includes singular fits; inspect fit flags before interpretation", fontsize=12)
    for suffix in ("png", "pdf"):
        fig.savefig(folder / f"random_effects_sensitivity.{suffix}", dpi=200)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(13, 6), constrained_layout=True)
    sensitivity_panel(axes[0], diagnostic_matrix("median_outlier_fraction"), "Median fraction of standardized residuals beyond +/-3")
    sensitivity_panel(axes[1], diagnostic_matrix("median_absolute_residual_fitted_rho"), "Median absolute-residual / fitted Spearman association", maximum=1)
    fig.suptitle("Residual screening across model/input-format cases\nDescriptive checks, not formal tests of model validity", fontsize=12)
    for suffix in ("png", "pdf"):
        fig.savefig(folder / f"residual_diagnostics.{suffix}", dpi=200)
    plt.close(fig)
    print(f"Wrote random-effects sensitivity summaries to {folder}; main results unchanged.")


if __name__ == "__main__":
    main()
