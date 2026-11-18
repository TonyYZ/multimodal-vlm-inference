"""Model-level regressions of human-likeness on model features (family, size, training data).

Companion to analyze_model_size.py, with the same layout and plotting style. One point per
model per study/format. Two outcomes, each run through the same fits and figures:

  inference      RMS distance between the model's and the humans' mean item-contrast
                 vectors (the Human column of each heatmap's ``*_distances.csv``;
                 lower = more human-like). Contrasts remove statement-level plausibility,
                 isolating the inference effect.
  comprehension  Pearson r between model and human raw ratings (higher = more
                 human-like). Animation: the 20 condition means of
                 model_human_raw_condition_agreement.csv (as in analyze_model_size.py);
                 gesture: condition means from per_item_scores.csv; emoji: all 190
                 judgments (emoji_pilot_agreement_summary.csv, "All").

Predictors:
  family             Qwen-VL (2.5/3), Qwen-Omni (2.5/3), Qwen3.8, Gemma 4, MiniCPM-o
  log2 size          total parameters from the model name (analyze_model_size.model_size_b)
  training tokens    pretraining tokens of the LLM backbone (Qwen2.5 tech report 18T,
                     Qwen3 tech report 36T; MiniCPM-o 4.5 is built on Qwen3-8B). Not
                     published for Gemma 4 or Qwen3.8, so those are left out of that fit.

Exploratory: n = 11 models (5 in gesture), mostly Qwen checkpoints.
"""
from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from analyze_model_size import fit_quadratic, model_key, model_size_b

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
DEFAULT_OUTPUT = RESULTS / "model_features"

DISTANCES = {
    "animation: split": RESULTS / "animation/inference/effect_profiles/item_effects/mean_item_contrasts_split_distances.csv",
    "animation: pure-video": RESULTS / "animation/inference/effect_profiles/item_effects/mean_item_contrasts_pure_video_distances.csv",
    "gesture: audio-video": RESULTS / "gesture/inference/effect_profiles/item_effects/mean_item_contrasts_audio_video_distances.csv",
    "emoji": RESULTS / "emoji/inference/effect_profiles/emoji_mean_item_contrasts_clustering_distances.csv",
}
FAMILY_ORDER = ["Qwen-VL", "Qwen3.8", "Qwen-Omni", "MiniCPM-o", "Gemma 4"]
# Backbone pretraining tokens (trillions); None = not published.
TRAINING_TOKENS_T = {"Qwen2.5": 18.0, "Qwen3": 36.0}
OUTCOMES = {
    "inference": {
        "column": "rms_distance", "stem": "distance",
        "axis": "RMS distance to humans\n(mean item contrasts; lower = more human-like)",
        "title": "Inference: mean item-contrast RMS distance to humans (lower = more human-like)",
    },
    "comprehension": {
        "column": "raw_r", "stem": "comprehension",
        "axis": "Model-human Pearson r, raw ratings\n(higher = more human-like)",
        "title": "Comprehension: correlation of model and human raw ratings (higher = more human-like)",
    },
}
ANIMATION_AGREEMENT = RESULTS / "animation/inference/scatter/model_human_raw_condition_agreement.csv"
GESTURE_SCORES = RESULTS / "gesture/human_tests/per_item_scores.csv"
EMOJI_AGREEMENT = RESULTS / "emoji/inference/comparison/emoji_pilot_agreement_summary.csv"


def write_csv(path: Path, rows: list[dict]) -> None:
    fields = list(dict.fromkeys(key for row in rows for key in row if not key.startswith("_")))
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def family(model: str) -> str:
    name = model.lower()
    if "qwen3.8" in name:
        return "Qwen3.8"
    if "-vl-" in name:
        return "Qwen-VL"
    if "omni" in name and "qwen" in name:
        return "Qwen-Omni"
    if "gemma" in name:
        return "Gemma 4"
    if "minicpm" in name:
        return "MiniCPM-o"
    raise ValueError(f"No family defined for {model}")


def training_tokens_t(model: str) -> float:
    name = model.lower()
    if "qwen2.5" in name:
        return TRAINING_TOKENS_T["Qwen2.5"]
    if "qwen3-" in name or "minicpm-o-4_5" in name:  # Qwen3-VL / Qwen3-Omni / MiniCPM-o 4.5 (Qwen3-8B)
        return TRAINING_TOKENS_T["Qwen3"]
    return math.nan  # Gemma 4, Qwen3.8: not published


def raw_correlations() -> dict[tuple[str, str], float]:
    """(cell, model) -> Pearson r between model and human raw ratings."""
    out = {}
    animation_formats = {"split": "animation: split", "video": "animation: pure-video"}
    with ANIMATION_AGREEMENT.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            if row["input_format"] in animation_formats:
                out[(animation_formats[row["input_format"]], row["model"])] = float(row["pearson_r"])
    with EMOJI_AGREEMENT.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            if row["experiment"] == "All":
                out[("emoji", row["model"])] = float(row["pearson_r"])
    cells = defaultdict(lambda: ([], []))  # condition-level means, matching the animation agreement table
    with GESTURE_SCORES.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            if row["model"] == "Human" or not row["score"] or not row["human_mean"] or "NA" in (row["score"], row["human_mean"]):
                continue
            key = (row["model"], row["condition"], row["environment"], row["premise"], row["inference"])
            cells[key][0].append(float(row["score"]))
            cells[key][1].append(float(row["human_mean"]))
    by_model = defaultdict(lambda: ([], []))
    for (model, *_), (scores, humans) in cells.items():
        by_model[model][0].append(np.mean(scores))
        by_model[model][1].append(np.mean(humans))
    for model, (scores, humans) in by_model.items():
        out[("gesture: audio-video", model)] = float(stats.pearsonr(scores, humans).statistic)
    return out


def collect_values() -> list[dict]:
    raw_r = raw_correlations()
    values = []
    for cell, path in DISTANCES.items():
        with path.open("r", encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                if row["model"] == "Human":
                    continue
                size = model_size_b(row["model"])
                values.append({
                    "cell": cell,
                    "model": row["model"],
                    "family": family(row["model"]),
                    "size_b": size,
                    "log2_size_b": math.log2(size),
                    "training_tokens_t": training_tokens_t(row["model"]),
                    "rms_distance": float(row["Human"]),
                    "raw_r": raw_r[(cell, row["model"])],
                })
    return values


def by_cell(values: list[dict]) -> dict[str, list[dict]]:
    groups = defaultdict(list)
    for row in values:
        groups[row["cell"]].append(row)
    return groups


def continuous_fit(rows: list[dict], key: str) -> dict:
    rows = [row for row in rows if np.isfinite(row[key])]
    x = np.array([row[key] for row in rows])
    y = np.array([row["y"] for row in rows])
    if len(rows) < 3 or np.ptp(x) == 0:
        return {"n_models": len(rows)}
    fit = stats.linregress(x, y)
    rank = stats.spearmanr(x, y)
    return {"n_models": len(rows), "slope": fit.slope, "intercept": fit.intercept, "r_squared": fit.rvalue**2,
            "p_value": fit.pvalue, "spearman_rho": rank.statistic, "spearman_p": rank.pvalue}


def family_fit(rows: list[dict]) -> dict:
    groups = defaultdict(list)
    for row in rows:
        groups[row["family"]].append(row["y"])
    y = np.array([row["y"] for row in rows])
    k, n = len(groups), len(rows)
    ss_between = sum(len(v) * (np.mean(v) - y.mean()) ** 2 for v in groups.values())
    ss_total = np.sum((y - y.mean()) ** 2)
    out = {"n_models": n, "n_families": k, "r_squared": ss_between / ss_total,
           "family_means": "; ".join(f"{f}={np.mean(groups[f]):.1f}" for f in FAMILY_ORDER if f in groups)}
    if k > 1 and n > k:  # one-way ANOVA: singleton families add a mean but no error df
        f_value = (ss_between / (k - 1)) / ((ss_total - ss_between) / (n - k))
        out |= {"f_value": f_value, "df": f"{k - 1}, {n - k}", "p_value": stats.f.sf(f_value, k - 1, n - k)}
    return out


def joint_fit(rows: list[dict], cell: str) -> list[dict]:
    """OLS with log2 size + training tokens together (models with known tokens only)."""
    rows = [row for row in rows if np.isfinite(row["training_tokens_t"])]
    x = np.column_stack([np.ones(len(rows)), [r["log2_size_b"] for r in rows], [r["training_tokens_t"] for r in rows]])
    y = np.array([r["y"] for r in rows])
    df = len(rows) - x.shape[1]
    if df < 1 or np.linalg.matrix_rank(x) < x.shape[1]:
        return []
    beta = np.linalg.lstsq(x, y, rcond=None)[0]
    resid = y - x @ beta
    se = np.sqrt(np.diag(resid @ resid / df * np.linalg.inv(x.T @ x)))
    r2 = 1 - resid @ resid / np.sum((y - y.mean()) ** 2)
    return [{"cell": cell, "term": term, "estimate": b, "std_error": s, "t_value": b / s,
             "p_value": 2 * stats.t.sf(abs(b / s), df), "n_models": len(rows), "r_squared": r2}
            for term, b, s in zip(["intercept", "log2_size_b", "training_tokens_t"], beta, se)]


def annotate(ax, rows: list[dict], numbers: dict[str, int], x_of) -> None:
    for row in rows:
        ax.annotate(str(numbers[row["model"]]), (x_of(row), row["y"]),
                    xytext=(4, 6 if numbers[row["model"]] % 2 == 0 else -12),
                    textcoords="offset points", fontsize=7)


def plot_continuous(values: list[dict], spec: dict, key: str, label: str, title: str, path: Path, log_x: bool) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    numbers = model_key(axes.flat[-1], values)
    for ax, (cell, rows) in zip(axes.flat, by_cell(values).items()):
        rows = [row for row in rows if np.isfinite(row[key])]
        x = np.array([row[key] for row in rows])
        y = np.array([row["y"] for row in rows])
        fit = continuous_fit(rows, "log2_size_b" if log_x else key)  # size is fitted per doubling
        ax.scatter(x, y, color="#4c78a8", s=45)
        annotate(ax, rows, numbers, lambda row: row[key])
        if "slope" in fit:
            line_x = np.geomspace(x.min(), x.max(), 100) if log_x else np.linspace(x.min(), x.max(), 100)
            line_v = np.log2(line_x) if log_x else line_x
            ax.plot(line_x, fit["intercept"] + fit["slope"] * line_v, color="#d95f02")
            unit = "doubling" if log_x else "trillion tokens"
            ax.set_title(f"{cell}\nSlope/{unit} = {fit['slope']:.2f}, p = {fit['p_value']:.3f}, "
                         f"R² = {fit['r_squared']:.2f}; n = {fit['n_models']}", fontsize=10)
        else:
            ax.set_title(f"{cell}\nToo few distinct values to fit; n = {fit['n_models']}", fontsize=10)
        if log_x:
            ax.set_xscale("log", base=2)
            ax.set_xticks([2, 4, 8, 16, 32], ["2B", "4B", "8B", "16B", "32B"])
        ax.set_xlabel(label)
        ax.set_ylabel(spec["axis"])
        ax.grid(alpha=0.25)
    for ax in list(axes.flat)[len(DISTANCES):]:  # unused panels and the key panel
        ax.axis("off")
    fig.suptitle(f"{title}\n{spec['title']}; exploratory", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, .94))
    for extension in ("png", "pdf"):
        fig.savefig(path.with_suffix(f".{extension}"), dpi=200)
    plt.close(fig)


def plot_family(values: list[dict], spec: dict, path: Path) -> None:
    """Horizontal dot plot: one row per family, one dot per model, grey range bar, orange mean."""
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    numbers = model_key(axes.flat[-1], values)
    for ax, (cell, rows) in zip(axes.flat, by_cell(values).items()):
        present = [f for f in FAMILY_ORDER if any(row["family"] == f for row in rows)]
        for y, f in enumerate(present):
            members = sorted((row for row in rows if row["family"] == f), key=lambda row: row["y"])
            x = [row["y"] for row in members]
            if len(x) > 1:
                ax.hlines(y, min(x), max(x), color="#bbbbbb", lw=2, zorder=1)
                ax.plot(np.mean(x), y, marker="|", color="#d95f02", markersize=22, mew=2.5, zorder=3)
            ax.scatter(x, [y] * len(x), color="#4c78a8", s=45, zorder=2)
            for i, row in enumerate(members):  # alternate labels above/below so nearby dots stay readable
                ax.annotate(str(numbers[row["model"]]), (row["y"], y),
                            xytext=(0, 8 if i % 2 == 0 else -14), textcoords="offset points", fontsize=7, ha="center")
        fit = family_fit(rows)
        test = f"F({fit['df']}) = {fit['f_value']:.2f}, p = {fit['p_value']:.3f}" if "p_value" in fit else "no error df"
        ax.set_title(f"{cell}\n{test}, R² = {fit['r_squared']:.2f}; n = {fit['n_models']}", fontsize=10)
        counts = {f: sum(row["family"] == f for row in rows) for f in present}
        ax.set_yticks(range(len(present)), [f"{f} (n = {counts[f]})" for f in present])
        ax.set_ylim(len(present) - .5, -.5)
        ax.set_xlabel(spec["axis"])
        ax.grid(axis="x", alpha=0.25)
    for ax in list(axes.flat)[len(DISTANCES):]:  # unused panels and the key panel
        ax.axis("off")
    fig.suptitle(f"Model family. {spec['title']}\nDots = models; grey = range within family; orange | = family mean "
                 "(n = 1 families are a single model); one-way ANOVA; exploratory", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, .94))
    for extension in ("png", "pdf"):
        fig.savefig(path.with_suffix(f".{extension}"), dpi=200)
    plt.close(fig)


def quadratic_rows(values: list[dict]) -> list[dict]:
    """Quadratic in centred log2 size (same method as analyze_model_size.fit_quadratic)."""
    out = []
    for cell, rows in by_cell(values).items():
        fit = fit_quadratic(np.array([r["log2_size_b"] for r in rows]), np.array([r["y"] for r in rows]))
        out.append({"cell": cell, "n_models": len(rows)} | fit)
    return out


def plot_size_quadratic(values: list[dict], spec: dict, fits: list[dict], path: Path) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    numbers = model_key(axes.flat[-1], values)
    for ax, (cell, rows), fit in zip(axes.flat, by_cell(values).items(), fits):
        x = np.array([row["size_b"] for row in rows])
        y = np.array([row["y"] for row in rows])
        line_x = np.geomspace(x.min(), x.max(), 100)
        centred = np.log2(line_x) - fit["center"]
        ax.scatter(x, y, color="#4c78a8", s=45)
        ax.plot(line_x, fit["intercept"] + fit["linear_coefficient"] * centred + fit["quadratic_coefficient"] * centred**2,
                color="#d95f02")
        annotate(ax, rows, numbers, lambda row: row["size_b"])
        turning = (f"Curve {fit['turning_point_type']} at {fit['turning_point_b']:.1f}B"
                   if fit["turning_point_in_observed_range"] else "Turning point outside observed range")
        ax.set_title(f"{cell}\nQuadratic p = {fit['quadratic_p']:.3f}, R² = {fit['quadratic_r_squared']:.2f}; "
                     f"n = {fit['n_models']}\n{turning}", fontsize=10)
        ax.set_xscale("log", base=2)
        ax.set_xticks([2, 4, 8, 16, 32], ["2B", "4B", "8B", "16B", "32B"])
        ax.set_xlabel("Model size (billions of parameters, log scale)")
        ax.set_ylabel(spec["axis"])
        ax.grid(alpha=0.25)
    for ax in list(axes.flat)[len(DISTANCES):]:
        ax.axis("off")
    fig.suptitle(f"Quadratic model-size analysis. {spec['title']}\nMid-sized models best = distance minimum / correlation "
                 "maximum inside the size range; exploratory", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, .94))
    for extension in ("png", "pdf"):
        fig.savefig(path.with_suffix(f".{extension}"), dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    values = collect_values()
    write_csv(args.output_dir / "model_feature_values.csv", values)
    univariate, joint, quadratic = [], [], []
    for outcome, spec in OUTCOMES.items():
        rows_o = [row | {"y": row[spec["column"]]} for row in values]
        tag = {"outcome": outcome}
        for cell, rows in by_cell(rows_o).items():
            univariate.append(tag | {"cell": cell, "predictor": "family"} | family_fit(rows))
            for key in ("log2_size_b", "training_tokens_t"):
                univariate.append(tag | {"cell": cell, "predictor": key} | continuous_fit(rows, key))
            joint += [tag | row for row in joint_fit(rows, cell)]
        fits = quadratic_rows(rows_o)
        quadratic += [tag | row for row in fits]
        stem = spec["stem"]
        plot_family(rows_o, spec, args.output_dir / f"model_family_{stem}")
        plot_continuous(rows_o, spec, "size_b", "Model size (billions of parameters, log scale)",
                        "Model size (total parameters)", args.output_dir / f"model_size_{stem}", log_x=True)
        plot_size_quadratic(rows_o, spec, fits, args.output_dir / f"model_size_{stem}_quadratic")
        plot_continuous(rows_o, spec, "training_tokens_t", "LLM backbone pretraining tokens (trillions)",
                        "Training-data size (backbone pretraining tokens; Gemma 4 and Qwen3.8 unpublished, omitted)",
                        args.output_dir / f"model_training_data_{stem}", log_x=False)
    write_csv(args.output_dir / "model_feature_univariate.csv", univariate)
    write_csv(args.output_dir / "model_feature_joint_size_training.csv", joint)
    write_csv(args.output_dir / "model_size_quadratic.csv", quadratic)
    print(f"Wrote model-feature analysis to {args.output_dir}")


if __name__ == "__main__":
    main()
