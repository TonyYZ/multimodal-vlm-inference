from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "results/animation/inference/comparison/presupposition_environment_model_averages.csv"
OUTPUT = ROOT / "results/animation/inference/comparison/presupposition_environment_model_averages"

ENVIRONMENTS = (
    ("Affirmative", "affirmative_delta"),
    ("Question", "question_delta"),
    ("Negation", "negation_delta"),
)
FORMATS = (
    ("pure-text", "Pure text", "#4C78A8"),
    ("pure-video", "Pure video", "#F2A65A"),
    ("split", "Split", "#59A14F"),
)


def main() -> None:
    with INPUT.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = {row["input_format"]: row for row in csv.DictReader(handle)}

    x = np.arange(len(ENVIRONMENTS))
    width = 0.24
    fig, ax = plt.subplots(figsize=(8.4, 5.2))
    for index, (mode, label, color) in enumerate(FORMATS):
        values = [float(rows[mode][field]) for _, field in ENVIRONMENTS]
        bars = ax.bar(
            x + (index - 1) * width,
            values,
            width,
            label=label,
            color=color,
        )
        ax.bar_label(bars, fmt="%.1f", padding=3, fontsize=8)

    ax.axhline(0, color="#333333", linewidth=1)
    ax.set_xticks(x, [label for label, _ in ENVIRONMENTS])
    ax.set_ylabel("Mean target - baseline rating")
    ax.set_xlabel("Presupposition environment")
    ax.set_title("Presupposition contrasts averaged across seven models")
    ax.grid(axis="y", color="#e5e5e5", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.legend(title="Input format", frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()

    for suffix in ("png", "pdf"):
        fig.savefig(OUTPUT.with_suffix(f".{suffix}"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {OUTPUT}.png and {OUTPUT}.pdf")


if __name__ == "__main__":
    main()
