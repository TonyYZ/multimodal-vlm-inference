"""Heatmaps of the main sampled-model tests (animation).

Left:  each source's own effect d (within-model LRT; Lyn's test for the human row).
Right: d_model - d_human (human-comparison LRT, the main test). A ≈ marks tests where the
       TOST equivalence test finds the model within +/- margin of humans.
Cells show the estimate and its LRT significance (*** p < .001, ** p < .01, * p < .05, ns).
Models failing the inclusion criteria (validity.csv) are labelled "excluded".
Inputs (the folder of --lrt, default results/animation/sampling_pilot): sampled_likelihood_ratio_tests.csv, effect_distributions.csv,
validity.csv. Output: sampled_test_heatmap.{png,pdf}.
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

from plot_model_barcharts import model_sort_key

PILOT = Path(__file__).resolve().parents[1] / "results" / "animation" / "sampling_pilot"
TESTS = [("Scalar Implicature", "Positive"), ("Scalar Implicature", "Negative"), ("Presupposition", "Question"),
         ("Presupposition", "None"), ("Supplement", "Unembedded"), ("Homogeneity", "Positive"),
         ("Homogeneity", "Negative")]
LABELS = ["SI\npositive", "SI\nnegative", "Presup.\nquestion", "Presup.\nnone", "Supplement\nunembedded",
          "Homog.\npositive", "Homog.\nnegative"]
# Diverging: blue (effect below 0 / weaker than humans) - neutral gray - red (above 0 / stronger).
CMAP = LinearSegmentedColormap.from_list("diverging", ["#2b6cb0", "#f0f0f0", "#c0392b"])


def stars(p):
    return "***" if p < .001 else "**" if p < .01 else "*" if p < .05 else "ns"


def key_rows(lrt):
    two_factor = lrt.condition.isin(["Scalar Implicature", "Supplement"])
    return lrt[(lrt.analysis == "human comparison") | (two_factor & (lrt.tested_effect == "Premise x inference")) |
               (~two_factor & (lrt.tested_effect == "Inference"))]


def row_label(source, excluded):
    model, _, mode = source.partition(" | ")
    label = model.split("/")[-1].replace("-Instruct", "") + (f"  ({mode})" if mode else "")
    return label + ("  — excluded" if source in excluded else "")


def draw(ax, values, pvalues, extra, rows, title, limit):
    norm = TwoSlopeNorm(vcenter=0, vmin=-limit, vmax=limit)
    image = ax.imshow(np.clip(values, -limit, limit), cmap=CMAP, norm=norm, aspect="auto")
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            if np.isnan(values[i, j]):
                continue
            strong = abs(values[i, j]) > .6 * limit
            ax.text(j, i, f"{values[i, j]:+.0f}{extra[i][j]}\n{stars(pvalues[i, j])}", ha="center", va="center",
                    fontsize=7.5, color="white" if strong else "#1a1a1a",
                    fontweight="bold" if pvalues[i, j] < .05 else "normal")
    ax.set_xticks(range(len(LABELS)), LABELS, fontsize=8)
    ax.set_yticks(range(len(rows)), rows, fontsize=8.5)
    ax.tick_params(length=0)
    ax.set_xticks(np.arange(-.5, len(LABELS)), minor=True)
    ax.set_yticks(np.arange(-.5, len(rows)), minor=True)
    ax.grid(which="minor", color="white", linewidth=2)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title(title, fontsize=10, loc="left")
    return image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lrt", type=Path, default=PILOT / "sampled_likelihood_ratio_tests.csv")
    parser.add_argument("--output", type=Path, default=PILOT / "sampled_test_heatmap")
    args = parser.parse_args()
    lrt = key_rows(pd.read_csv(args.lrt, keep_default_na=False))
    validity = pd.read_csv(args.lrt.parent / "validity.csv", keep_default_na=False)
    excluded = {m.rsplit(" | T=", 1)[0] for m in validity.loc[validity.included.astype(str) != "True", "model"]}
    effects = pd.read_csv(args.lrt.parent / "effect_distributions.csv", keep_default_na=False)
    effects["source"] = effects.model.str.rsplit(" | T=", n=1).str[0]
    margin = effects.margin.iloc[0]
    equivalent = {(r.source, r.condition, r.environment): str(r.equivalent) == "True" for r in effects.itertuples()}

    models = sorted(set(lrt.model) - {"Human"}, key=lambda s: (s.split(" | ")[1] != "split", model_sort_key(s)))
    within = lrt[lrt.analysis == "within"].set_index(["model", "condition", "environment"])
    comparison = lrt[lrt.analysis == "human comparison"].set_index(["model", "condition", "environment"])
    grid = lambda frame, sources, column: np.array(
        [[frame.loc[(s, *t), column] if (s, *t) in frame.index else np.nan for t in TESTS] for s in sources])

    sources = ["Human", *models]
    fig, (left, right) = plt.subplots(1, 2, figsize=(15, 7.2), dpi=180,
                                      gridspec_kw=dict(width_ratios=[1, 1], wspace=.08))
    image = draw(left, grid(within, sources, "d_scale_estimate"), grid(within, sources, "p_value"),
                 [[""] * len(TESTS)] * len(sources), [row_label(s, excluded) for s in sources],
                 "Effect d (within-source LRT)", 160)
    fig.colorbar(image, ax=left, orientation="horizontal", fraction=.04, pad=.12, label="d (rating points)")
    mark = [["≈" if equivalent.get((s, *t)) else "" for t in TESTS] for s in models]
    values = np.vstack([np.full(len(TESTS), np.nan), grid(comparison, models, "d_scale_estimate")])
    pvalues = np.vstack([np.ones(len(TESTS)), grid(comparison, models, "p_value")])
    image = draw(right, values, pvalues, [[""] * len(TESTS), *mark], [""] * len(sources),
                 f"Model − human: d_model − d_human (main test; ≈ equivalent within ±{margin:g})", 120)
    right.set_yticks([])
    right.text(3, 0, "reference row (humans)", ha="center", va="center", fontsize=8, color="#888888")
    fig.colorbar(image, ax=right, orientation="horizontal", fraction=.04, pad=.12,
                 label="d_model − d_human (blue: weaker than humans, red: stronger)")
    fig.suptitle("Animation study, temperature samples (T = 1, n = 99): LRT results per test", fontsize=12, x=.02,
                 ha="left")
    for fmt in ("png", "pdf"):
        fig.savefig(args.output.with_suffix(f".{fmt}"), bbox_inches="tight")
    print(f"Wrote {args.output}.png/.pdf")


if __name__ == "__main__":
    main()
