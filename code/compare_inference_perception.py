from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr, rankdata, spearmanr


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EFFECTS = ROOT / "results/model_comparison/model_human_summary.csv"
DEFAULT_RAW = ROOT / "results/model_human_scatter/model_human_raw_condition_agreement.csv"
DEFAULT_PERCEPTION = (
    ROOT / "results/description_judge_charts/description_judge_model_summary.csv"
)
DEFAULT_OUTPUT_DIR = ROOT / "results/inference_perception_comparison"

METRICS = (
    ("mean_absolute_error", "Effect MAE", True),
    ("mae", "Raw-condition MAE", True),
    ("pearson_r", "Pearson r", False),
    ("spearman_rho", "Spearman rho", False),
)
CAPABILITY_METRICS = (
    "Effect human likeness",
    "Raw-rating human likeness",
    "Human-pattern Pearson r",
    "Human-pattern Spearman rho",
)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def short_name(model: str) -> str:
    return model.split("/")[-1].replace("-Instruct", "").replace("-it", "")


def split_rows(path: Path, mode_field: str) -> dict[str, dict[str, str]]:
    return {
        row.get("model", row.get("raw_model", "")): row
        for row in read_csv(path)
        if row[mode_field] == "split"
    }


def merge_rows(effect_path: Path, raw_path: Path, perception_path: Path) -> list[dict]:
    effects = split_rows(effect_path, "input_mode")
    raw = split_rows(raw_path, "input_format")
    perception = split_rows(perception_path, "input_mode")
    models = sorted(set(effects) & set(raw) & set(perception))
    if not models:
        raise ValueError("No models occur in all three split-format summaries.")

    rows = []
    for model in models:
        row = {
            "model": model,
            "perception_visual_score": float(perception[model]["mean_visual_score"]),
            **{field: float(raw[model][field]) for field, _, _ in METRICS[1:]},
            "mean_absolute_error": float(effects[model]["mean_absolute_error"]),
        }
        rows.append(row)

    for field, _, lower_is_better in (
        ("perception_visual_score", "Perception", False),
        *METRICS,
    ):
        values = np.array([row[field] for row in rows])
        ranks = rankdata(values if lower_is_better else -values, method="average")
        for row, rank in zip(rows, ranks):
            row[f"{field}_rank"] = float(rank)
    return rows


def write_rows(rows: list[dict], path: Path) -> None:
    fields = (
        "model",
        "perception_visual_score",
        "perception_visual_score_rank",
        "mean_absolute_error",
        "mean_absolute_error_rank",
        "mae",
        "mae_rank",
        "pearson_r",
        "pearson_r_rank",
        "spearman_rho",
        "spearman_rho_rank",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda row: row["perception_visual_score_rank"]))


def write_associations(rows: list[dict], path: Path) -> None:
    perception = np.array([row["perception_visual_score"] for row in rows])
    output = []
    for field, label, lower_is_better in METRICS:
        inference = np.array([row[field] for row in rows])
        result = spearmanr(perception, -inference if lower_is_better else inference)
        output.append(
            {
                "inference_metric": label,
                "n_models": len(rows),
                "rank_agreement_rho": result.statistic,
                "p_value": result.pvalue,
            }
        )
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=output[0].keys())
        writer.writeheader()
        writer.writerows(output)


def matched_mode_associations(
    effect_path: Path, raw_path: Path, perception_path: Path
) -> list[dict]:
    effects = read_csv(effect_path)
    raw = read_csv(raw_path)
    perception = read_csv(perception_path)
    output = []

    for mode, raw_mode in (("pure-video", "video"), ("split", "split")):
        visual = {
            row["model"]: float(row["mean_visual_score"])
            for row in perception
            if row["input_mode"] == mode
        }
        effect_likeness = {
            row["raw_model"]: 100.0 - float(row["mean_absolute_error"])
            for row in effects
            if row["input_mode"] == mode
        }
        raw_metrics = {
            row["model"]: {
                "Raw-rating human likeness": 100.0 - float(row["mae"]),
                "Human-pattern Pearson r": float(row["pearson_r"]),
                "Human-pattern Spearman rho": float(row["spearman_rho"]),
            }
            for row in raw
            if row["input_format"] == raw_mode
        }
        models = sorted(set(visual) & set(effect_likeness) & set(raw_metrics))
        metrics = {
            "Effect human likeness": [effect_likeness[model] for model in models],
            **{
                metric: [raw_metrics[model][metric] for model in models]
                for metric in CAPABILITY_METRICS[1:]
            },
        }
        visual_scores = [visual[model] for model in models]

        for metric, inference_scores in metrics.items():
            linear = pearsonr(visual_scores, inference_scores)
            rank = spearmanr(visual_scores, inference_scores)
            output.append(
                {
                    "input_mode": mode,
                    "inference_metric": metric,
                    "n_models": len(models),
                    "pearson_r": linear.statistic,
                    "pearson_p": linear.pvalue,
                    "spearman_rho": rank.statistic,
                    "spearman_p": rank.pvalue,
                }
            )
    return output


def write_dicts(rows: list[dict], path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_correlation_heatmap(rows: list[dict], output_dir: Path) -> None:
    modes = ("pure-video", "split")
    lookup = {(row["input_mode"], row["inference_metric"]): row for row in rows}
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), constrained_layout=True)

    for ax, field, p_field, title in (
        (axes[0], "pearson_r", "pearson_p", "Pearson correlation"),
        (axes[1], "spearman_rho", "spearman_p", "Spearman correlation"),
    ):
        matrix = np.array(
            [[lookup[(mode, metric)][field] for metric in CAPABILITY_METRICS] for mode in modes]
        )
        image = ax.imshow(matrix, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
        ax.set_xticks(range(len(CAPABILITY_METRICS)), CAPABILITY_METRICS, rotation=25, ha="right")
        ax.set_yticks(range(len(modes)), modes)
        ax.set_title(title)
        for y, mode in enumerate(modes):
            for x, metric in enumerate(CAPABILITY_METRICS):
                row = lookup[(mode, metric)]
                ax.text(
                    x,
                    y,
                    f"r = {row[field]:.2f}\np = {row[p_field]:.3f}",
                    ha="center",
                    va="center",
                    fontsize=8,
                )
    fig.colorbar(image, ax=axes, label="Visual–inference correlation", shrink=0.85)
    fig.suptitle("Association between visual capability and human-like inference")
    for extension in ("png", "pdf"):
        fig.savefig(output_dir / f"visual_inference_correlations.{extension}", dpi=200)
    plt.close(fig)


def plot_scatter(rows: list[dict], output_dir: Path) -> None:
    perception = np.array([row["perception_visual_score"] for row in rows])
    fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))

    for ax, (field, label, lower_is_better) in zip(axes.flat, METRICS):
        inference = np.array([row[field] for row in rows])
        rho = spearmanr(perception, -inference if lower_is_better else inference).statistic
        ax.scatter(perception, inference, s=55, color="#4c78a8")
        for row in rows:
            ax.annotate(
                short_name(row["model"]),
                (row["perception_visual_score"], row[field]),
                xytext=(4, 4),
                textcoords="offset points",
                fontsize=7,
            )
        ax.set_xlabel("Perception: visual description score (higher is better)")
        ax.set_ylabel(f"Inference: {label}")
        direction = "lower is better" if lower_is_better else "higher is better"
        ax.set_title(f"{label} ({direction})\nrank agreement rho = {rho:.2f}")
        ax.grid(alpha=0.25)

    fig.suptitle("Split-format perception and inference performance", fontsize=14)
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(output_dir / f"split_inference_perception_scatter.{suffix}", dpi=200)
    plt.close(fig)


def plot_ranks(rows: list[dict], output_dir: Path) -> None:
    rows = sorted(rows, key=lambda row: row["perception_visual_score_rank"])
    columns = (
        ("perception_visual_score_rank", "Perception"),
        ("mean_absolute_error_rank", "Effect MAE"),
        ("mae_rank", "Raw MAE"),
        ("pearson_r_rank", "Pearson r"),
        ("spearman_rho_rank", "Spearman rho"),
    )
    matrix = np.array([[row[field] for field, _ in columns] for row in rows])
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    image = ax.imshow(matrix, cmap="RdYlGn_r", vmin=1, vmax=len(rows), aspect="auto")
    ax.set_xticks(range(len(columns)), [label for _, label in columns])
    ax.set_yticks(range(len(rows)), [short_name(row["model"]) for row in rows])
    for y in range(len(rows)):
        for x in range(len(columns)):
            ax.text(x, y, f"{matrix[y, x]:.0f}", ha="center", va="center")
    ax.set_title("Split-format model ranks (1 = best)")
    fig.colorbar(image, ax=ax, label="Rank")
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(output_dir / f"split_inference_perception_ranks.{suffix}", dpi=200)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare split-format inference performance with perception scores."
    )
    parser.add_argument("--effects", type=Path, default=DEFAULT_EFFECTS)
    parser.add_argument("--raw", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--perception", type=Path, default=DEFAULT_PERCEPTION)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = merge_rows(args.effects, args.raw, args.perception)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_rows(rows, args.output_dir / "split_inference_perception_comparison.csv")
    write_associations(rows, args.output_dir / "split_inference_perception_associations.csv")
    associations = matched_mode_associations(args.effects, args.raw, args.perception)
    write_dicts(
        associations,
        args.output_dir / "visual_inference_correlations.csv",
    )
    plot_correlation_heatmap(associations, args.output_dir)
    plot_scatter(rows, args.output_dir)
    plot_ranks(rows, args.output_dir)
    print(f"Wrote comparison outputs to {args.output_dir}")


if __name__ == "__main__":
    main()
