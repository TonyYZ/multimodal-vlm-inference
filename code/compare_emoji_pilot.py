from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr, spearmanr

from compare_model_performance import parse_score


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = PROJECT_ROOT / "materials" / "emoji_study" / "emoji_pilot_manifest.csv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results" / "emoji" / "inference" / "comparison"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare emoji-pilot model ratings with matched human means."
    )
    parser.add_argument("ratings", type=Path, nargs="+")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--require-complete", action="store_true",
                        help="Refuse to replace analyses if any model lacks a valid original-item rating.")
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"No rows to write: {path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def compare_posttext_controls(joined, ratings, manifest_path, output_dir):
    """Keep new premise controls separate from the original human comparison."""
    controls = load_csv(manifest_path)
    rows, effects = [], []
    for original in joined:
        if original["experiment"] != "Experiment 3":
            continue
        control = next(r for r in controls if r["target_source"] == original["source_video"]
                       and r["statement_type"] == original["statement_type"])
        response = ratings.get((original["model"], control["source_video"], control["statement_type"]), {})
        for premise, material, score, human_mean, human_n, error in (
            ("TargetPremise", original, original["model_score"], original["human_mean"], original["human_n"], ""),
            ("ControlPremise", control, parse_score(response.get("response")), "", 0,
             response.get("error", "" if response else "missing response")),
        ):
            rows.append({"model": original["model"], "emoji": material["emoji"],
                         "premise": premise, "statement_type": material["statement_type"],
                         "source_video": material["source_video"], "sentence": material["sentence"],
                         "inference": material["inference"], "model_score": score,
                         "human_mean": human_mean, "human_n": human_n,
                         "human_available": premise == "TargetPremise", "error": error})
    for model, emoji in sorted({(r["model"], r["emoji"]) for r in rows}):
        scores = {(r["premise"], r["statement_type"]): r["model_score"]
                  for r in rows if r["model"] == model and r["emoji"] == emoji}
        if any(value is None for value in scores.values()):
            continue
        target = scores["TargetPremise", "target"] - scores["TargetPremise", "baseline"]
        control = scores["ControlPremise", "target"] - scores["ControlPremise", "baseline"]
        effects.append({"model": model, "emoji": emoji, "target_premise_delta": target,
                        "control_premise_delta": control, "premise_contrast": target - control,
                        "interpretation": "descriptive contrast only; new human controls unavailable"})
    write_csv(output_dir / "posttext_premise_controls.csv", rows)
    if effects:
        write_csv(output_dir / "posttext_premise_contrasts.csv", effects)


def correlation(values_x: list[float], values_y: list[float], fn) -> tuple[float, float]:
    if len(values_x) < 3 or len(set(values_x)) < 2 or len(set(values_y)) < 2:
        return float("nan"), float("nan")
    result = fn(values_x, values_y)
    return float(result.statistic), float(result.pvalue)


def effect_rows(rows: list[dict]) -> list[dict]:
    effects = []

    exp1 = [row for row in rows if row["experiment"] == "Experiment 1"]
    exp1_groups: dict[tuple[str, str, str], dict[str, dict]] = {}
    for row in exp1:
        key = (row["model"], row["emoji"], row["environment"], row["inference_type"])
        exp1_groups.setdefault(key, {})[row["material_condition"]] = row
    for (_, emoji, environment, inference_type), pair in exp1_groups.items():
        if {"Target", "Control"} <= pair.keys():
            effects.append(
                make_effect_row(
                    experiment="Experiment 1",
                    emoji=emoji,
                    environment=environment,
                    contrast="target_minus_control_material",
                    target=pair["Target"],
                    control=pair["Control"],
                    inference_type=inference_type,
                )
            )

    for experiment in ("Experiment 2", "Experiment 3"):
        groups: dict[tuple[str, str], dict[str, dict]] = {}
        for row in rows:
            if row["experiment"] == experiment:
                groups.setdefault((row["model"], row["source_video"]), {})[
                    row["statement_type"]
                ] = row
        for pair in groups.values():
            if {"target", "baseline"} <= pair.keys():
                effects.append(
                    make_effect_row(
                        experiment=experiment,
                        emoji=pair["target"]["emoji"],
                        environment=pair["target"]["environment"],
                        contrast="target_minus_control_inference",
                        target=pair["target"],
                        control=pair["baseline"],
                        inference_type="Target minus control",
                    )
                )
    return effects


def make_effect_row(
    *,
    experiment: str,
    emoji: str,
    environment: str,
    contrast: str,
    target: dict,
    control: dict,
    inference_type: str,
) -> dict:
    human_target = float(target["human_mean"])
    human_control = float(control["human_mean"])
    model_target = float(target["model_score"])
    model_control = float(control["model_score"])
    human_effect = human_target - human_control
    model_effect = model_target - model_control
    return {
        "model": target["model"],
        "experiment": experiment,
        "emoji": emoji,
        "environment": environment,
        "inference_type": inference_type,
        "contrast": contrast,
        "human_target": human_target,
        "human_control": human_control,
        "human_effect": human_effect,
        "model_target": model_target,
        "model_control": model_control,
        "model_effect": model_effect,
        "effect_error": model_effect - human_effect,
        "absolute_effect_error": abs(model_effect - human_effect),
    }


def plot_comparison(rows: list[dict], effects: list[dict], path: Path) -> None:
    colors = {
        "Experiment 1": "#2A9D8F",
        "Experiment 2": "#E9C46A",
        "Experiment 3": "#E76F51",
    }
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.4))

    for experiment, color in colors.items():
        subset = [row for row in rows if row["experiment"] == experiment]
        axes[0].scatter(
            [float(row["human_mean"]) for row in subset],
            [float(row["model_score"]) for row in subset],
            color=color,
            label=experiment,
            alpha=0.85,
            edgecolor="white",
            linewidth=0.5,
        )
    axes[0].plot([0, 100], [0, 100], color="#555555", linestyle="--", linewidth=1)
    axes[0].set(xlim=(0, 100), ylim=(0, 100), xlabel="Human mean", ylabel="Model rating")
    axes[0].set_title("Raw judgments")
    axes[0].legend(frameon=False, fontsize=8)

    experiments = list(colors)
    human_effects = [
        np.mean([float(row["human_effect"]) for row in effects if row["experiment"] == exp])
        for exp in experiments
    ]
    model_effects = [
        np.mean([float(row["model_effect"]) for row in effects if row["experiment"] == exp])
        for exp in experiments
    ]
    x = np.arange(len(experiments))
    width = 0.36
    axes[1].bar(x - width / 2, human_effects, width, color="#9E9E9E", label="Human")
    axes[1].bar(x + width / 2, model_effects, width, color="#2A9D8F", label="Model")
    axes[1].axhline(0, color="#555555", linewidth=0.8)
    axes[1].set_xticks(x, ["Co-text", "Pro-text", "Post-text"])
    axes[1].set_ylabel("Mean target-control effect")
    axes[1].set_title("Experimental effects")
    axes[1].legend(frameon=False)

    fig.suptitle("Emoji inference pilot: Qwen2.5-VL-3B vs. humans", fontsize=13)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rating_rows = [row for path in args.ratings for row in load_jsonl(path)]
    ratings = {}
    for row in rating_rows:
        key = row["model"], row["source_video"], row["statement_type"]
        # Preserve raw attempts, but never let an invalid attempt replace a usable rating.
        if key not in ratings or parse_score(row.get("response")) is not None:
            ratings[key] = row
    models = sorted({row["model"] for row in rating_rows})
    manifest = load_csv(args.manifest)
    if args.require_complete:
        missing = [(model_id, human["source_video"], human["statement_type"])
                   for model_id in models for human in manifest
                   if parse_score(ratings.get((model_id, human["source_video"], human["statement_type"]), {}).get("response")) is None]
        if missing:
            raise ValueError(f"Incomplete ratings: {len(missing)} missing/invalid cells; first: {missing[0]}")
    joined = []
    for model_id in models:
        for human in manifest:
            model = ratings.get(
                (model_id, human["source_video"], human["statement_type"])
            )
            score = parse_score(model.get("response") if model else None)
            if score is None:
                continue
            human_mean = float(human["human_mean"])
            joined.append(
                {
                    **human,
                    "model": model_id,
                    "model_score": score,
                    "absolute_error": abs(score - human_mean),
                }
            )

    detail_path = args.output_dir / "emoji_pilot_model_human.csv"
    for model_id in models:
        count = sum(row["model"] == model_id for row in joined)
        print(f"Usable original ratings: {model_id}: {count}/{len(manifest)}")
    write_csv(detail_path, joined)
    if any(source.endswith("_ControlPremise.txt") for _, source, _ in ratings):
        compare_posttext_controls(joined, ratings,
                                 args.manifest.with_name("posttext_controls_manifest.csv"), args.output_dir)

    summaries = []
    groups = [
        (model, experiment)
        for model in models
        for experiment in (*sorted({row["experiment"] for row in joined}), "All")
    ]
    for model, experiment in groups:
        model_rows = [row for row in joined if row["model"] == model]
        rows = model_rows if experiment == "All" else [
            row for row in model_rows if row["experiment"] == experiment
        ]
        if not rows:
            continue
        human_values = [float(row["human_mean"]) for row in rows]
        model_values = [float(row["model_score"]) for row in rows]
        pearson_r, pearson_p = correlation(model_values, human_values, pearsonr)
        spearman_rho, spearman_p = correlation(model_values, human_values, spearmanr)
        summaries.append(
            {
                "model": model,
                "experiment": experiment,
                "n_judgments": len(rows),
                "human_mean": np.mean(human_values),
                "model_mean": np.mean(model_values),
                "mae": np.mean([float(row["absolute_error"]) for row in rows]),
                "pearson_r": pearson_r,
                "pearson_p": pearson_p,
                "spearman_rho": spearman_rho,
                "spearman_p": spearman_p,
            }
        )

    summary_path = args.output_dir / "emoji_pilot_agreement_summary.csv"
    write_csv(summary_path, summaries)

    effects = effect_rows(joined)
    effects_path = args.output_dir / "emoji_pilot_effects.csv"
    write_csv(effects_path, effects)
    effect_summaries = []
    for model in models:
        for experiment in ["Experiment 1", "Experiment 2", "Experiment 3", "All"]:
            model_rows = [row for row in effects if row["model"] == model]
            rows = model_rows if experiment == "All" else [
                row for row in model_rows if row["experiment"] == experiment
            ]
            if not rows:
                continue
            human_values = [float(row["human_effect"]) for row in rows]
            model_values = [float(row["model_effect"]) for row in rows]
            pearson_r, pearson_p = correlation(model_values, human_values, pearsonr)
            spearman_rho, spearman_p = correlation(model_values, human_values, spearmanr)
            effect_summaries.append(
                {
                    "model": model,
                    "experiment": experiment,
                    "n_effects": len(rows),
                    "human_effect_mean": np.mean(human_values),
                    "model_effect_mean": np.mean(model_values),
                    "effect_mae": np.mean(
                        [float(row["absolute_effect_error"]) for row in rows]
                    ),
                    "pearson_r": pearson_r,
                    "pearson_p": pearson_p,
                    "spearman_rho": spearman_rho,
                    "spearman_p": spearman_p,
                }
            )
    effect_summary_path = args.output_dir / "emoji_pilot_effect_summary.csv"
    write_csv(effect_summary_path, effect_summaries)
    figure_path = args.output_dir / "emoji_pilot_human_comparison.png"
    if len(models) == 1:
        plot_comparison(joined, effects, figure_path)

    print(f"Matched judgments: {len(joined)}")
    print(f"Wrote details to {detail_path}")
    print(f"Wrote summary to {summary_path}")
    print(f"Wrote effects to {effects_path}")
    print(f"Wrote effect summary to {effect_summary_path}")
    if len(models) == 1:
        print(f"Wrote comparison figure to {figure_path}")


if __name__ == "__main__":
    main()
