"""Bar charts of temperature-sampled ratings in the style of plot_model_barcharts.py.

Each dot is one participant's story-averaged rating for a cell: an individual human
participant in the Human panel, one temperature sample (pseudo-participant) in model panels.
Bars and error bars are the mean and standard error across those participants.
Above each test, the LRT results from sampled_lrt.R are printed:
  d  = the source's own effect (within-model LRT; Lyn's test for humans)
  Δ  = d_model - d_human (human-comparison LRT, the main test)
with *** p < .001, ** p < .01, * p < .05, ns otherwise. Models failing the inclusion
criteria (validity.csv) are marked in their panel title.
Input: pseudo_participant_ratings.csv written by sampled_human_comparison.py.
"""
import argparse
from collections import defaultdict
from pathlib import Path

import pandas as pd

import plot_model_barcharts as bars
from plot_model_barcharts import plot_combined_input_mode, plot_model

ROOT = Path(__file__).resolve().parents[1]
PILOT = ROOT / "results" / "animation" / "sampling_pilot"
DEFAULT_INPUT = PILOT / "pseudo_participant_ratings.csv"
DEFAULT_OUTPUT = ROOT / "results" / "animation" / "inference" / "bar_charts_samples"
FAMILY = {"Scalar Implicature": "ScalarImplicature1", "Presupposition": "Presupposition1",
          "Supplement": "Supplement1", "Homogeneity": "Homogeneity1"}
GROUP = {"Scalar implicature": "Scalar Implicature"}  # bar-chart group name -> condition in the LRT file


def participant_rows(frame):
    """One row per source x participant x cell, in the row format plot_bars expects."""
    means = frame.groupby(["source", "participant", "condition", "environment", "GesturalPremise",
                           "InferenceType"]).value.mean().unstack("InferenceType").reset_index()
    rows = []
    for row in means.to_dict("records"):
        model, _, mode = row["source"].partition(" | ")
        rows.append({"raw_model": model, "model": model if model == "Human" else f"{model} {mode}",
                     "input_mode": mode or "human", "task_family": FAMILY[row["condition"]],
                     "condition": row["environment"], "premise": f"{row['GesturalPremise']}Premise",
                     "target_score": "" if pd.isna(row.get("Target")) else str(row["Target"]),
                     "baseline_score": "" if pd.isna(row.get("Baseline")) else str(row["Baseline"])})
    return rows


def stars(p):
    return "***" if p < .001 else "**" if p < .01 else "*" if p < .05 else "ns"


def main_tests(path):
    """(source, condition, environment) -> {analysis: (d, p)} for each test's key effect."""
    lrt = pd.read_csv(path, keep_default_na=False)
    two_factor = lrt.condition.isin(["Scalar Implicature", "Supplement"])
    key = (lrt.analysis == "human comparison") | (two_factor & (lrt.tested_effect == "Premise x inference")) | \
          (~two_factor & (lrt.tested_effect == "Inference"))
    tests = defaultdict(dict)
    for row in lrt[key].itertuples():
        tests[(row.model, row.condition, row.environment)][row.analysis] = (row.d_scale_estimate, row.p_value)
    return tests


def annotating(plot_bars, tests, excluded):
    """Wraps plot_bars so every panel also shows its LRT results above each test."""
    def wrapper(ax, rows, title, *args, **kwargs):
        compact = kwargs.get("compact_headers", args[3] if len(args) > 3 else False)
        excluded_here = rows and rows[0]["raw_model"] != "Human" and \
            f"{rows[0]['raw_model']} | {rows[0]['input_mode']}" in excluded
        plotted = plot_bars(ax, rows, title + ("   [excluded: fails literal check]" if excluded_here else ""),
                            *args, **kwargs)
        if not plotted:
            return plotted
        source = "Human" if rows[0]["raw_model"] == "Human" else f"{rows[0]['raw_model']} | {rows[0]['input_mode']}"
        specs = bars.group_spec()
        spans = defaultdict(list)
        for i, spec in enumerate(specs):
            spans[(GROUP.get(spec["group"], spec["group"]), spec["condition"])].append(i)
        ax.set_ylim(-4, 128)
        font = 7 if compact else 9
        for (condition, environment), positions in spans.items():
            result = tests.get((source, condition, environment), {})
            x = sum(positions) / len(positions)
            if "within" in result:
                d, p = result["within"]
                ax.text(x, 120, f"d {d:+.0f}{stars(p)}", ha="center", va="center", fontsize=font, color="#333333")
            if "human comparison" in result:
                d, p = result["human comparison"]
                ax.text(x, 109, f"Δ {d:+.0f}{stars(p)}", ha="center", va="center", fontsize=font,
                        color="#111111" if p < .05 else "#888888", fontweight="bold" if p < .05 else "normal")
        return plotted
    return wrapper


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--lrt", type=Path, default=PILOT / "sampled_likelihood_ratio_tests.csv")
    parser.add_argument("--validity", type=Path, default=PILOT / "validity.csv")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--formats", nargs="+", default=["png", "pdf"], choices=["png", "pdf", "svg"])
    parser.add_argument("--models-per-page", type=int, default=3)
    args = parser.parse_args()
    validity = pd.read_csv(args.validity, keep_default_na=False)
    excluded = {m.rsplit(" | T=", 1)[0] for m in validity.loc[validity.included.astype(str) != "True", "model"]}
    bars.plot_bars = annotating(bars.plot_bars, main_tests(args.lrt), excluded)
    rows = participant_rows(pd.read_csv(args.input, keep_default_na=False))
    human = [row for row in rows if row["raw_model"] == "Human"]
    by_mode, by_model = defaultdict(list), defaultdict(list)
    for row in rows:
        if row["raw_model"] != "Human":
            by_mode[row["input_mode"]].append(row)
            by_model[row["model"]].append(row)
    for model, model_rows in sorted(by_model.items()):
        plot_model(model, model_rows, args.output_dir, tuple(args.formats))
    plot_model("Human", human, args.output_dir, tuple(args.formats))
    for mode, mode_rows in by_mode.items():
        plot_combined_input_mode(mode, mode_rows, human, args.output_dir, tuple(args.formats), args.models_per_page)
    print(f"Wrote charts to {args.output_dir}")


if __name__ == "__main__":
    main()
