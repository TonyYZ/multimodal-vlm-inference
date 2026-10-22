from __future__ import annotations

import argparse
import csv
import math
import re
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EFFECTS = ROOT / "results/model_comparison/model_human_summary.csv"
DEFAULT_RAW = ROOT / "results/model_human_scatter/model_human_raw_condition_agreement.csv"
DEFAULT_PERCEPTION = ROOT / "results/description_judge_charts/description_judge_model_summary.csv"
DEFAULT_OUTPUT = ROOT / "results/model_size_analysis"

METRIC_LABELS = {
    "perception_visual_score": "Visual perception score",
    "effect_human_likeness": "Effect human likeness",
    "raw_human_likeness": "Raw-rating human likeness",
    "raw_pearson_r": "Model-human Pearson r",
    "raw_spearman_rho": "Model-human Spearman rho",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def model_size_b(model: str) -> float:
    matches = re.findall(r"(\d+(?:\.\d+)?)B", model, flags=re.IGNORECASE)
    if not matches:
        raise ValueError(f"Cannot infer parameter count from model name: {model}")
    return float(matches[-1])


def add_value(rows: list[dict], model: str, mode: str, metric: str, value: float) -> None:
    size = model_size_b(model)
    rows.append(
        {
            "model": model,
            "size_b": size,
            "log2_size_b": math.log2(size),
            "input_mode": mode,
            "metric": metric,
            "value": value,
        }
    )


def collect_values(effect_path: Path, raw_path: Path, perception_path: Path) -> list[dict]:
    values = []
    for row in read_csv(effect_path):
        add_value(
            values,
            row["raw_model"],
            row["input_mode"],
            "effect_human_likeness",
            100.0 - float(row["mean_absolute_error"]),
        )

    mode_names = {"text": "pure-text", "video": "pure-video", "split": "split"}
    for row in read_csv(raw_path):
        mode = mode_names[row["input_format"]]
        add_value(values, row["model"], mode, "raw_human_likeness", 100.0 - float(row["mae"]))
        add_value(values, row["model"], mode, "raw_pearson_r", float(row["pearson_r"]))
        add_value(values, row["model"], mode, "raw_spearman_rho", float(row["spearman_rho"]))

    for row in read_csv(perception_path):
        add_value(
            values,
            row["model"],
            row["input_mode"],
            "perception_visual_score",
            float(row["mean_visual_score"]),
        )
    return values


def regressions(values: list[dict]) -> list[dict]:
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in values:
        groups[(row["input_mode"], row["metric"])].append(row)

    output = []
    for (mode, metric), rows in sorted(groups.items()):
        if len(rows) < 3:
            continue
        log_size = np.array([row["log2_size_b"] for row in rows])
        size = np.array([row["size_b"] for row in rows])
        score = np.array([row["value"] for row in rows])
        linear = stats.linregress(log_size, score)
        rank = stats.spearmanr(size, score)
        output.append(
            {
                "input_mode": mode,
                "metric": metric,
                "n_models": len(rows),
                "slope_per_size_doubling": linear.slope,
                "intercept": linear.intercept,
                "r_squared": linear.rvalue**2,
                "pearson_r": linear.rvalue,
                "pearson_p": linear.pvalue,
                "spearman_rho": rank.statistic,
                "spearman_p": rank.pvalue,
            }
        )
    return output


def fit_quadratic(log_size: np.ndarray, score: np.ndarray) -> dict:
    centered = log_size - log_size.mean()
    linear_x = np.column_stack((np.ones(len(score)), centered))
    quadratic_x = np.column_stack((linear_x, centered**2))
    linear_beta = np.linalg.lstsq(linear_x, score, rcond=None)[0]
    beta = np.linalg.lstsq(quadratic_x, score, rcond=None)[0]
    linear_sse = np.sum((score - linear_x @ linear_beta) ** 2)
    residuals = score - quadratic_x @ beta
    sse = np.sum(residuals**2)
    degrees_freedom = len(score) - quadratic_x.shape[1]
    covariance = (sse / degrees_freedom) * np.linalg.inv(quadratic_x.T @ quadratic_x)
    quadratic_se = np.sqrt(covariance[2, 2])
    quadratic_t = beta[2] / quadratic_se
    quadratic_p = 2 * stats.t.sf(abs(quadratic_t), degrees_freedom)
    total_sse = np.sum((score - score.mean()) ** 2)
    r_squared = 1 - sse / total_sse
    adjusted_r_squared = 1 - (1 - r_squared) * (len(score) - 1) / degrees_freedom
    improvement_f = ((linear_sse - sse) / 1) / (sse / degrees_freedom)
    improvement_p = stats.f.sf(improvement_f, 1, degrees_freedom)
    turning_log_size = log_size.mean() - beta[1] / (2 * beta[2])
    turning_size = 2**turning_log_size
    turning_type = "maximum" if beta[2] < 0 else "minimum"
    in_range = log_size.min() <= turning_log_size <= log_size.max()
    return {
        "linear_coefficient": beta[1],
        "quadratic_coefficient": beta[2],
        "quadratic_t": quadratic_t,
        "quadratic_p": quadratic_p,
        "linear_vs_quadratic_f": improvement_f,
        "linear_vs_quadratic_p": improvement_p,
        "quadratic_r_squared": r_squared,
        "quadratic_adjusted_r_squared": adjusted_r_squared,
        "turning_point_b": turning_size,
        "turning_point_type": turning_type,
        "turning_point_in_observed_range": in_range,
        "intercept": beta[0],
        "center": log_size.mean(),
    }


def quadratic_regressions(values: list[dict]) -> list[dict]:
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in values:
        groups[(row["input_mode"], row["metric"])].append(row)

    output = []
    for (mode, metric), rows in sorted(groups.items()):
        if len(rows) < 4:
            continue
        fit = fit_quadratic(
            np.array([row["log2_size_b"] for row in rows]),
            np.array([row["value"] for row in rows]),
        )
        output.append(
            {
                "input_mode": mode,
                "metric": metric,
                "n_models": len(rows),
                **{key: value for key, value in fit.items() if key not in {"intercept", "center"}},
            }
        )
    return output


def short_name(model: str) -> str:
    return model.split("/")[-1].replace("-Instruct", "").replace("-it", "")


def plot_split(values: list[dict], output_dir: Path) -> None:
    split = [row for row in values if row["input_mode"] == "split"]
    by_metric = defaultdict(list)
    for row in split:
        by_metric[row["metric"]].append(row)

    fig, axes = plt.subplots(2, 3, figsize=(13, 8.5))
    for ax, metric in zip(axes.flat, METRIC_LABELS):
        rows = by_metric[metric]
        x = np.array([row["size_b"] for row in rows])
        log_x = np.log2(x)
        y = np.array([row["value"] for row in rows])
        fit = stats.linregress(log_x, y)
        rank = stats.spearmanr(x, y)
        line_x = np.geomspace(x.min(), x.max(), 100)

        ax.scatter(x, y, color="#4c78a8", s=45)
        ax.plot(line_x, fit.intercept + fit.slope * np.log2(line_x), color="#d95f02")
        for row in rows:
            ax.annotate(
                short_name(row["model"]),
                (row["size_b"], row["value"]),
                xytext=(4, 4),
                textcoords="offset points",
                fontsize=7,
            )
        ax.set_xscale("log", base=2)
        ax.set_xticks([2, 4, 8, 16, 32], ["2B", "4B", "8B", "16B", "32B"])
        ax.set_xlabel("Model size (billions of parameters, log scale)")
        ax.set_ylabel(METRIC_LABELS[metric])
        ax.set_title(
            f"Slope/doubling = {fit.slope:.2f}, p = {fit.pvalue:.3f}\n"
            f"Size rho = {rank.statistic:.2f}, p = {rank.pvalue:.3f}"
        )
        ax.grid(alpha=0.25)

    axes.flat[-1].axis("off")
    fig.suptitle("Model size and split-format performance", fontsize=14)
    fig.tight_layout()
    for extension in ("png", "pdf"):
        fig.savefig(output_dir / f"model_size_split_performance.{extension}", dpi=200)
    plt.close(fig)


def plot_split_quadratic(values: list[dict], output_dir: Path) -> None:
    split = [row for row in values if row["input_mode"] == "split"]
    by_metric = defaultdict(list)
    for row in split:
        by_metric[row["metric"]].append(row)

    fig, axes = plt.subplots(2, 3, figsize=(13, 8.5))
    for ax, metric in zip(axes.flat, METRIC_LABELS):
        rows = by_metric[metric]
        x = np.array([row["size_b"] for row in rows])
        log_x = np.log2(x)
        y = np.array([row["value"] for row in rows])
        fit = fit_quadratic(log_x, y)
        line_x = np.geomspace(x.min(), x.max(), 100)
        centered_line = np.log2(line_x) - fit["center"]
        line_y = (
            fit["intercept"]
            + fit["linear_coefficient"] * centered_line
            + fit["quadratic_coefficient"] * centered_line**2
        )

        ax.scatter(x, y, color="#4c78a8", s=45)
        ax.plot(line_x, line_y, color="#d95f02")
        for row in rows:
            ax.annotate(
                short_name(row["model"]),
                (row["size_b"], row["value"]),
                xytext=(4, 4),
                textcoords="offset points",
                fontsize=7,
            )
        ax.set_xscale("log", base=2)
        ax.set_xticks([2, 4, 8, 16, 32], ["2B", "4B", "8B", "16B", "32B"])
        ax.set_xlabel("Model size (billions of parameters, log scale)")
        ax.set_ylabel(METRIC_LABELS[metric])
        turning = (
            f"{fit['turning_point_type'].capitalize()} = {fit['turning_point_b']:.1f}B"
            if fit["turning_point_in_observed_range"]
            else "Turning point outside observed range"
        )
        ax.set_title(f"Quadratic p = {fit['quadratic_p']:.3f}\n{turning}")
        ax.grid(alpha=0.25)

    axes.flat[-1].axis("off")
    fig.suptitle("Quadratic model-size analysis for split-format performance", fontsize=14)
    fig.tight_layout()
    for extension in ("png", "pdf"):
        fig.savefig(output_dir / f"model_size_split_quadratic.{extension}", dpi=200)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze model-level associations between parameter count and performance."
    )
    parser.add_argument("--effects", type=Path, default=DEFAULT_EFFECTS)
    parser.add_argument("--raw", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--perception", type=Path, default=DEFAULT_PERCEPTION)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    values = collect_values(args.effects, args.raw, args.perception)
    results = regressions(values)
    quadratic_results = quadratic_regressions(values)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "model_size_performance_values.csv", values)
    write_csv(args.output_dir / "model_size_performance_regressions.csv", results)
    write_csv(args.output_dir / "model_size_quadratic_regressions.csv", quadratic_results)
    plot_split(values, args.output_dir)
    plot_split_quadratic(values, args.output_dir)
    print(f"Wrote model-size analysis to {args.output_dir}")


if __name__ == "__main__":
    main()
