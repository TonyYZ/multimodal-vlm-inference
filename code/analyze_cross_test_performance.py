from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from statistics import mean

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from item_effect_metrics import DEFAULT_EFFECTS, effect_rows, summarize_effects


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RAW_POINTS = ROOT / "results/animation/inference/scatter/model_human_scatter_points.csv"
DEFAULT_OUTPUT = ROOT / "results/animation/analysis/cross_test"
TESTS = ("Scalar implicature", "Homogeneity", "Presupposition", "Supplement")
MODES = ("pure-text", "pure-video", "split")
SEMANTIC_FAMILIES = (
    ("Alternatives/exhaustification", "Scalar implicature", "Homogeneity"),
    ("Projection", "Presupposition", "Supplement"),
)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: list[dict[str, str]]) -> list[dict]:
    import pandas as pd
    summary = summarize_effects(pd.DataFrame(rows), by_family=True).rename(columns={"inference_group": "test"})
    summary["negative_effect_distance"] = -summary.effect_rms_distance
    return summary[np.isfinite(summary.effect_rms_distance)].to_dict("records")


def summarize_raw(rows: list[dict[str, str]]) -> list[dict]:
    grouped: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    for row in rows:
        error = abs(float(row["model_endorsement"]) - float(row["human_endorsement"]))
        grouped[(row["raw_model"], row["input_mode"], row["inference_group"])].append(error)

    output = []
    for (model, mode, test), errors in grouped.items():
        mae = mean(errors)
        output.append(
            {
                "model": model,
                "input_mode": mode,
                "test": test,
                "n_conditions": len(errors),
                "mean_absolute_error": mae,
                "human_likeness": 100.0 - mae,
            }
        )
    return output


def score_lookup(rows: list[dict]) -> dict[tuple[str, str, str], float]:
    return {
        (row["input_mode"], row["model"], row["test"]): row.get("negative_effect_distance", row.get("human_likeness"))
        for row in rows
    }


def complete_models(rows: list[dict], mode: str) -> list[str]:
    by_model: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        if row["input_mode"] == mode:
            by_model[row["model"]].add(row["test"])
    return sorted(model for model, tests in by_model.items() if set(TESTS) <= tests)


def correlation_stats(left: list[float], right: list[float], *, exact: bool = True) -> dict:
    """Constant scores have undefined correlations; bound permutation work at 9999."""
    if len(left) < 3 or not np.isfinite([left, right]).all() or np.ptp(left) == 0 or np.ptp(right) == 0:
        return dict.fromkeys(("pearson_r", "pearson_p", "spearman_rho", "spearman_p"), np.nan) | {"p_method": "undefined (constant profile)"}
    method = stats.PermutationMethod(n_resamples=9999, random_state=42) if exact else None
    linear, rank = stats.pearsonr(left, right, method=method), stats.spearmanr(left, right)
    # SciPy enumerates exactly when possible, otherwise uses seeded resampling.
    rank_p = stats.pearsonr(stats.rankdata(left), stats.rankdata(right), method=method).pvalue if exact else rank.pvalue
    return {"pearson_r": linear.statistic, "pearson_p": linear.pvalue,
            "spearman_rho": rank.statistic, "spearman_p": rank_p,
            "p_method": "pairing permutations (SciPy; up to 9999; seed 42)" if exact else "asymptotic (SciPy)"}


def adjusted_correlations(rows: list[dict]) -> np.ndarray:
    p_values = np.array([row["spearman_p"] for row in rows])
    valid = np.isfinite(p_values)
    adjusted = stats.false_discovery_control(np.where(valid, p_values, 1), method="bh")
    return np.where(valid, adjusted, np.nan)


def rankings(rows: list[dict]) -> list[dict]:
    lookup = score_lookup(rows)
    output = []
    for mode in MODES:
        models = complete_models(rows, mode)
        test_ranks = {}
        for test in TESTS:
            scores = np.array([lookup[(mode, model, test)] for model in models])
            test_ranks[test] = stats.rankdata(-scores, method="average")

        for index, model in enumerate(models):
            row = {"model": model, "input_mode": mode}
            for test in TESTS:
                row[f"{test}_score"] = lookup[(mode, model, test)]
                row[f"{test}_rank"] = test_ranks[test][index]
            row["mean_rank"] = mean(row[f"{test}_rank"] for test in TESTS)
            output.append(row)

        mode_rows = [row for row in output if row["input_mode"] == mode]
        overall_ranks = stats.rankdata(
            [row["mean_rank"] for row in mode_rows], method="average"
        )
        for row, rank in zip(mode_rows, overall_ranks):
            row["overall_rank"] = rank
    return sorted(output, key=lambda row: (MODES.index(row["input_mode"]), row["overall_rank"]))


def inter_test_correlations(rows: list[dict]) -> list[dict]:
    lookup = score_lookup(rows)
    output = []
    for mode in MODES:
        models = complete_models(rows, mode)
        mode_rows = []
        for left, right in combinations(TESTS, 2):
            left_scores = [lookup[(mode, model, left)] for model in models]
            right_scores = [lookup[(mode, model, right)] for model in models]
            mode_rows.append(
                {
                    "input_mode": mode,
                    "test_1": left,
                    "test_2": right,
                    "n_models": len(models),
                    **correlation_stats(left_scores, right_scores),
                }
            )
        adjusted = adjusted_correlations(mode_rows)
        for row, q_value in zip(mode_rows, adjusted):
            row["spearman_q"] = q_value
        output.extend(mode_rows)
    return output


def generalization_correlations(rows: list[dict]) -> list[dict]:
    lookup = score_lookup(rows)
    output = []
    for mode in MODES:
        models = complete_models(rows, mode)
        matrix = np.array(
            [[lookup[(mode, model, test)] for test in TESTS] for model in models]
        )
        mode_rows = []
        for index, test in enumerate(TESTS):
            target = matrix[:, index]
            other_tests = matrix[:, np.arange(len(TESTS)) != index].mean(axis=1)
            mode_rows.append(
                {
                    "input_mode": mode,
                    "target_test": test,
                    "n_models": len(models),
                    **correlation_stats(target, other_tests),
                }
            )
        adjusted = adjusted_correlations(mode_rows)
        for row, q_value in zip(mode_rows, adjusted):
            row["spearman_q"] = q_value
        output.extend(mode_rows)
    return output


def semantic_family_correlations(
    effect_performance: list[dict], raw_performance: list[dict]
) -> list[dict]:
    output = []
    for level, performance in (("effect", effect_performance), ("raw", raw_performance)):
        lookup = score_lookup(performance)
        level_rows = []
        for mode in MODES:
            models = complete_models(performance, mode)
            for family, left, right in SEMANTIC_FAMILIES:
                left_scores = [lookup[(mode, model, left)] for model in models]
                right_scores = [lookup[(mode, model, right)] for model in models]
                level_rows.append(
                    {
                        "analysis_level": level,
                        "input_mode": mode,
                        "semantic_family": family,
                        "test_1": left,
                        "test_2": right,
                        "n_models": len(models),
                        **correlation_stats(left_scores, right_scores, exact=level == "effect"),
                    }
                )
        adjusted = adjusted_correlations(level_rows)
        for row, q_value in zip(level_rows, adjusted):
            row["spearman_q"] = q_value
        output.extend(level_rows)
    return output


def plot_semantic_families(rows: list[dict], output: Path) -> None:
    colors = {"Alternatives/exhaustification": "#4c78a8", "Projection": "#e45756"}
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), sharey=True, constrained_layout=True)
    for ax, level in zip(axes, ("effect", "raw")):
        level_rows = [row for row in rows if row["analysis_level"] == level]
        x = np.arange(len(MODES))
        width = 0.34
        for index, (family, _, _) in enumerate(SEMANTIC_FAMILIES):
            family_rows = {
                row["input_mode"]: row
                for row in level_rows
                if row["semantic_family"] == family
            }
            positions = x + (index - 0.5) * width
            values = [family_rows[mode]["spearman_rho"] for mode in MODES]
            bars = ax.bar(positions, values, width, color=colors[family], label=family)
            labels = [f"p={family_rows[mode]['spearman_p']:.3f}" if np.isfinite(family_rows[mode]["spearman_p"]) else "" for mode in MODES]
            ax.bar_label(bars, labels=labels, padding=3, fontsize=8)
            for position, value in zip(positions, values):
                if not np.isfinite(value):
                    ax.text(position, 0.04, "NA", ha="center", fontsize=9)
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_xticks(x, MODES)
        ax.set_ylim(-1.15, 1.15)
        ax.set_title("Mean item-contrast alignment" if level == "effect" else "Raw-rating human likeness")
        ax.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("Within-family Spearman correlation")
    axes[1].legend(title="Proposed semantic family", frameon=False, loc="lower left")
    fig.suptitle("Related semantic families: exploratory human-alignment associations\nEffects: negative RMS item-contrast distance; raw: 100 minus rating MAE", fontsize=12)
    for extension in ("png", "pdf"):
        fig.savefig(output / f"semantic_family_correlations.{extension}", dpi=200)
    plt.close(fig)


def short_name(model: str) -> str:
    return model.split("/")[-1].replace("-Instruct", "").replace("-it", "")


def plot_mode(
    performance: list[dict], ranks: list[dict], correlations: list[dict], mode: str, output: Path
) -> None:
    lookup = score_lookup(performance)
    mode_ranks = sorted(
        (row for row in ranks if row["input_mode"] == mode),
        key=lambda row: row["overall_rank"],
    )
    models = [row["model"] for row in mode_ranks]
    raw = -np.array([[lookup[(mode, model, test)] for test in TESTS] for model in models])

    rho = np.eye(len(TESTS))
    p_values = np.zeros((len(TESTS), len(TESTS)))
    for row in correlations:
        if row["input_mode"] != mode:
            continue
        left, right = TESTS.index(row["test_1"]), TESTS.index(row["test_2"])
        rho[left, right] = rho[right, left] = row["spearman_rho"]
        p_values[left, right] = p_values[right, left] = row["spearman_p"]

    fig, axes = plt.subplots(1, 2, figsize=(13, 6), constrained_layout=True)
    performance_image = axes[0].imshow(raw, cmap="YlOrRd", vmin=0, aspect="auto")
    axes[0].set_xticks(range(len(TESTS)), TESTS, rotation=25, ha="right")
    axes[0].set_yticks(range(len(models)), [short_name(model) for model in models])
    axes[0].set_title("RMS item-contrast distance from humans\nLower is closer, not a significance test")
    for y in range(len(models)):
        for x in range(len(TESTS)):
            axes[0].text(x, y, f"{raw[y, x]:.1f}", ha="center", va="center", fontsize=8)
    fig.colorbar(performance_image, ax=axes[0], label="RMS distance (rating points)")

    for i in range(len(TESTS)):
        if np.ptp(raw[:, i]) == 0:
            rho[i, i] = np.nan
    cmap = plt.get_cmap("RdBu_r").copy()
    cmap.set_bad("#d9d9d9")
    correlation_image = axes[1].imshow(rho, cmap=cmap, vmin=-1, vmax=1)
    axes[1].set_xticks(range(len(TESTS)), TESTS, rotation=25, ha="right")
    axes[1].set_yticks(range(len(TESTS)), TESTS)
    axes[1].set_title("Inter-test Spearman correlations")
    for y in range(len(TESTS)):
        for x in range(len(TESTS)):
            label = "NA" if not np.isfinite(rho[y, x]) else (
                f"{rho[y, x]:.2f}" if x == y else f"{rho[y, x]:.2f}\np={p_values[y, x]:.3f}"
            )
            axes[1].text(x, y, label, ha="center", va="center", fontsize=8)
    fig.colorbar(correlation_image, ax=axes[1], label="Spearman rho")
    fig.suptitle(f"Cross-test item-contrast alignment: {mode}\nExploratory; incomplete contrasts excluded, not nonsignificant tests", fontsize=12)

    slug = mode.replace("-", "_")
    for extension in ("png", "pdf"):
        fig.savefig(output / f"cross_test_{slug}.{extension}", dpi=200)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Test whether models that are human-like on one phenomenon are human-like on others."
    )
    parser.add_argument("--effects", type=Path, default=DEFAULT_EFFECTS)
    parser.add_argument("--raw-points", type=Path, default=DEFAULT_RAW_POINTS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    performance = summarize(effect_rows(args.effects).to_dict("records"))
    raw_performance = summarize_raw(read_csv(args.raw_points))
    rank_rows = rankings(performance)
    correlation_rows = inter_test_correlations(performance)
    generalization_rows = generalization_correlations(performance)
    family_rows = semantic_family_correlations(performance, raw_performance)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "cross_test_performance.csv", performance)
    write_csv(args.output_dir / "cross_test_rankings.csv", rank_rows)
    write_csv(args.output_dir / "inter_test_correlations.csv", correlation_rows)
    write_csv(args.output_dir / "test_generalization_correlations.csv", generalization_rows)
    write_csv(args.output_dir / "semantic_family_correlations.csv", family_rows)
    for mode in MODES:
        plot_mode(performance, rank_rows, correlation_rows, mode, args.output_dir)
    plot_semantic_families(family_rows, args.output_dir)
    print(f"Wrote cross-test analysis to {args.output_dir}")


if __name__ == "__main__":
    main()
