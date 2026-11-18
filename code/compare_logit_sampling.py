"""Robustness check: exact rating distributions (from logits) vs temperature samples (animation).

For each model, compares per-cell expected ratings with the mean of its 99 samples, and the
per-test effect d computed from each. Monte Carlo error of a sampled mean is sd / sqrt(n).
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from sampled_human_comparison import CELLS, KEY, OUT, contrasts, human_ratings, model_ratings


def exact_ratings(path):
    cells = pd.read_csv(CELLS, keep_default_na=False).drop_duplicates(["source_video", "statement_type"])
    frame = pd.DataFrame([json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()])
    frame["source_video"] = frame.source_video.str.replace("\\", "/").str.split("/").str[-1]
    return frame.merge(cells[["source_video", "statement_type", *KEY]], on=["source_video", "statement_type"],
                       validate="one_to_one")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("logit_files", nargs="+", type=Path)
    parser.add_argument("--sampled-suffix", default="_T1_n99",
                        help="Suffix of the sampled file to compare against (e.g. _T1_n99_penalty1.05).")
    parser.add_argument("--tag", default="", help="Appended to output file names.")
    args = parser.parse_args()
    human_d = contrasts(human_ratings()).groupby(["condition", "environment"]).d.mean().rename("human_d")
    cell_rows, effect_rows = [], []
    for path in args.logit_files:
        exact = exact_ratings(path)
        sampled_path = path.with_name(path.name.replace("_logit_scores", args.sampled_suffix))
        sampled = model_ratings(sampled_path)
        sampled = sampled[sampled.value.notna()]
        name = f"{exact.model.iloc[0].split('/')[-1]} | {exact.input_mode.iloc[0]}"
        stats = sampled.groupby(KEY).value.agg(sampled_mean="mean", sampled_sd="std", n="size").reset_index()
        merged = exact[[*KEY, "expected", "sd", "mass"]].merge(stats, on=KEY, validate="one_to_one")
        merged["difference"] = merged.sampled_mean - merged.expected
        merged["monte_carlo_se"] = merged.sampled_sd / np.sqrt(merged.n)
        cell_rows.append(merged.assign(model=name))
        exact_d = contrasts(exact.assign(participant="exact", value=exact.expected)).set_index(
            ["condition", "environment"]).d.rename("exact_d")
        sampled_d = contrasts(sampled).groupby(["condition", "environment"]).d.mean().rename("sampled_d")
        effects = pd.concat([exact_d, sampled_d, human_d], axis=1).reset_index().assign(model=name)
        effects["sampled_minus_exact"] = effects.sampled_d - effects.exact_d
        effect_rows.append(effects)
    cells, effects = pd.concat(cell_rows), pd.concat(effect_rows)
    cells.to_csv(OUT / f"logit_vs_sampled_cells{args.tag}.csv", index=False)
    effects.to_csv(OUT / f"logit_vs_sampled_effects{args.tag}.csv", index=False)
    summary = cells.groupby("model").apply(lambda g: pd.Series({
        "r_cells": np.corrcoef(g.expected, g.sampled_mean)[0, 1],
        "mean_abs_diff": g.difference.abs().mean(), "max_abs_diff": g.difference.abs().max(),
        "mean_monte_carlo_se": g.monte_carlo_se.mean(),
        "share_within_2se": (g.difference.abs() <= 2 * g.monte_carlo_se).mean(),
        "min_number_mass": g.mass.min(), "mean_number_mass": g.mass.mean()}), include_groups=False)
    summary.to_csv(OUT / f"logit_vs_sampled_summary{args.tag}.csv")
    pd.set_option("display.width", 220)
    print(summary.round(3).to_string())
    print(effects.round(1).to_string(index=False))


if __name__ == "__main__":
    main()
