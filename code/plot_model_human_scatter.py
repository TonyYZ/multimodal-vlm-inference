from __future__ import annotations

import argparse
import csv
import re
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
from scipy import stats


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RATINGS = PROJECT_ROOT / "results" / "model_comparison" / "ratings_paired.csv"
DEFAULT_HUMAN = PROJECT_ROOT / "materials" / "stats" / "TSC_EXP_2_human_effects.csv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results" / "model_human_scatter"

INFERENCE_ORDER = [
    ("Scalar implication", ("ScalarImplicature1", "ScalarImplicature2", "ScalarImplicature3")),
    ("Presupposition", ("Presupposition1", "Presupposition2")),
    ("Supplement", ("Supplement1", "Supplement2")),
    ("Homogeneity", ("Homogeneity1", "Homogeneity2")),
]
CONDITION_ORDER = {
    "Scalar implication": ("Positive", "Negative"),
    "Presupposition": ("Question", "None"),
    "Supplement": ("Unembedded",),
    "Homogeneity": ("Positive", "Negative"),
}
PREMISE_ORDER = ("TargetPremise", "ControlPremise")
STATEMENT_TYPES = ("target", "baseline")
INPUT_MODE_ORDER = ("pure-text", "pure-video", "split")
INPUT_MODE_LABELS = {
    "pure-text": "Text",
    "pure-video": "Video",
    "split": "Split",
}
INPUT_FORMAT_LABELS = {
    "pure-text": "text",
    "pure-video": "video",
    "split": "split",
}
HUMAN_SCORE_COLUMNS = {
    ("TargetPremise", "target"): "Target_Target",
    ("TargetPremise", "baseline"): "Target_Baseline",
    ("ControlPremise", "target"): "Control_Target",
    ("ControlPremise", "baseline"): "Control_Baseline",
}
STATEMENT_LABELS = {
    "target": "Target",
    "baseline": "Baseline",
}
PREMISE_LABELS = {
    "TargetPremise": "Target premise",
    "ControlPremise": "Control premise",
}
COLORS = {
    "target": "#69c98e",
    "baseline": "#909090",
}
MARKERS = {
    "Scalar implication": "s",
    "Presupposition": "o",
    "Supplement": "^",
    "Homogeneity": "D",
}


def model_slug(model: str) -> str:
    slug = model.lower().replace("/", "__")
    slug = re.sub(r"[^a-z0-9_.-]+", "_", slug)
    return slug.strip("_")


def display_model_name(model: str) -> str:
    return model.split("/")[-1]


def load_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def numeric(value: object) -> float | None:
    if value in ("", None, "NA"):
        return None
    try:
        return float(value)
    except ValueError:
        return None


def inference_group(task_family: str) -> str:
    for label, prefixes in INFERENCE_ORDER:
        if task_family.startswith(prefixes):
            return label
    return "Other"


def condition_label(group: str, condition: str) -> str:
    if group == "Supplement":
        return "Unembedded"
    return condition


def human_lookup(rows: list[dict]) -> dict[tuple[str, str], dict]:
    lookup = {}
    for row in rows:
        condition = row["Condition"]
        environment = row["Environment"]
        if condition == "Scalar Implicature":
            condition = "Scalar implication"
        lookup[(condition, environment)] = row
    return lookup


def mean(values: list[float]) -> float:
    return sum(values) / len(values)


def has_variance(values: list[float]) -> bool:
    return len(set(values)) > 1


def pearson_result(xs: list[float], ys: list[float]) -> tuple[float | None, float | None]:
    if len(xs) < 2 or len(xs) != len(ys):
        return None, None
    if not has_variance(xs) or not has_variance(ys):
        return None, None
    result = stats.pearsonr(xs, ys)
    return float(result.statistic), float(result.pvalue)


def spearman_result(xs: list[float], ys: list[float]) -> tuple[float | None, float | None]:
    if len(xs) < 2 or len(xs) != len(ys):
        return None, None
    if not has_variance(xs) or not has_variance(ys):
        return None, None
    result = stats.spearmanr(xs, ys)
    return float(result.statistic), float(result.pvalue)


def linear_regression(xs: list[float], ys: list[float]) -> tuple[float, float] | None:
    if len(xs) < 2 or len(xs) != len(ys):
        return None
    x_mean = mean(xs)
    y_mean = mean(ys)
    x_var = sum((x - x_mean) ** 2 for x in xs)
    if x_var == 0:
        return None
    slope = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys)) / x_var
    intercept = y_mean - slope * x_mean
    return slope, intercept


def fmt(value: float | None) -> str:
    if value is None:
        return ""
    return f"{value:.6f}"


def build_points(rating_rows: list[dict], human_rows: list[dict]) -> list[dict]:
    human_by_cell = human_lookup(human_rows)
    grouped: dict[tuple[str, str, str, str, str, str], list[float]] = defaultdict(list)

    for row in rating_rows:
        group = inference_group(row["task_family"])
        if group == "Other":
            continue
        condition = condition_label(group, row["condition"])
        premise = row["premise"]
        for statement_type, score_column in (
            ("target", "target_score"),
            ("baseline", "baseline_score"),
        ):
            score = numeric(row[score_column])
            if score is None:
                continue
            grouped[
                (
                    row["raw_model"],
                    row["input_mode"],
                    group,
                    condition,
                    premise,
                    statement_type,
                )
            ].append(score)

    points = []
    for key, model_scores in sorted(grouped.items()):
        raw_model, input_mode, group, condition, premise, statement_type = key
        if group in {"Presupposition", "Homogeneity"} and premise != "TargetPremise":
            continue
        human_row = human_by_cell.get((group, condition))
        if not human_row:
            continue
        human_column = HUMAN_SCORE_COLUMNS.get((premise, statement_type))
        if not human_column:
            continue
        human_score = numeric(human_row.get(human_column))
        if human_score is None:
            continue

        points.append(
            {
                "raw_model": raw_model,
                "input_mode": input_mode,
                "inference_group": group,
                "condition": condition,
                "premise": premise,
                "statement_type": statement_type,
                "model_endorsement": f"{mean(model_scores):.3f}",
                "human_endorsement": f"{human_score:.3f}",
                "n_model_videos": len(model_scores),
            }
        )
    return points


def points_by_model_mode(points: list[dict]) -> dict[tuple[str, str], list[dict]]:
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for point in points:
        grouped[(point["raw_model"], point["input_mode"])].append(point)
    return grouped


def agreement_rows(points: list[dict]) -> list[dict]:
    rows = []
    for (raw_model, input_mode), group_points in sorted(points_by_model_mode(points).items()):
        xs = [numeric(point["model_endorsement"]) for point in group_points]
        ys = [numeric(point["human_endorsement"]) for point in group_points]
        pairs = [(x, y) for x, y in zip(xs, ys) if x is not None and y is not None]
        xs = [x for x, _ in pairs]
        ys = [y for _, y in pairs]
        n = len(pairs)
        mae = mean([abs(x - y) for x, y in pairs]) if pairs else None
        pearson, pearson_p = pearson_result(xs, ys)
        spearman, spearman_p = spearman_result(xs, ys)
        rows.append(
            {
                "model": raw_model,
                "input_format": INPUT_FORMAT_LABELS.get(input_mode, input_mode),
                "n_conditions": n,
                "mae": fmt(mae),
                "pearson_r": fmt(pearson),
                "pearson_p": fmt(pearson_p),
                "spearman_rho": fmt(spearman),
                "spearman_p": fmt(spearman_p),
            }
        )
    return rows


def largest_error_rows(points: list[dict], limit: int = 5) -> list[dict]:
    rows = []
    for (raw_model, input_mode), group_points in sorted(points_by_model_mode(points).items()):
        scored = []
        for point in group_points:
            model_score = numeric(point["model_endorsement"])
            human_score = numeric(point["human_endorsement"])
            if model_score is None or human_score is None:
                continue
            scored.append((abs(model_score - human_score), model_score, human_score, point))
        scored.sort(key=lambda item: item[0], reverse=True)
        for rank, (abs_error, model_score, human_score, point) in enumerate(scored[:limit], start=1):
            rows.append(
                {
                    "model": raw_model,
                    "input_format": INPUT_FORMAT_LABELS.get(input_mode, input_mode),
                    "rank": rank,
                    "inference_group": point["inference_group"],
                    "condition": point["condition"],
                    "premise": point["premise"],
                    "statement_type": point["statement_type"],
                    "model_endorsement": f"{model_score:.3f}",
                    "human_endorsement": f"{human_score:.3f}",
                    "absolute_error": f"{abs_error:.3f}",
                    "n_model_videos": point["n_model_videos"],
                }
            )
    return rows


def sort_key(point: dict) -> tuple:
    group_order = {group: i for i, (group, _) in enumerate(INFERENCE_ORDER)}
    condition_order = {
        group: {condition: i for i, condition in enumerate(conditions)}
        for group, conditions in CONDITION_ORDER.items()
    }
    return (
        point["raw_model"],
        INPUT_MODE_ORDER.index(point["input_mode"])
        if point["input_mode"] in INPUT_MODE_ORDER
        else len(INPUT_MODE_ORDER),
        group_order.get(point["inference_group"], 99),
        condition_order.get(point["inference_group"], {}).get(point["condition"], 99),
        PREMISE_ORDER.index(point["premise"]) if point["premise"] in PREMISE_ORDER else 99,
        STATEMENT_TYPES.index(point["statement_type"])
        if point["statement_type"] in STATEMENT_TYPES
        else 99,
    )


def plot_model(raw_model: str, points: list[dict], output_dir: Path, formats: tuple[str, ...]) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.6), dpi=180, sharex=True, sharey=True)
    fig.suptitle(display_model_name(raw_model), fontsize=13, y=0.98)

    for ax, input_mode in zip(axes, INPUT_MODE_ORDER):
        panel_points = [p for p in points if p["input_mode"] == input_mode]
        ax.set_title(INPUT_MODE_LABELS[input_mode], fontsize=11)
        ax.plot([0, 100], [0, 100], color="#c9c9c9", linewidth=1.0, linestyle="--", zorder=1)
        ax.set_xlim(-4, 104)
        ax.set_ylim(-4, 104)
        ax.set_xticks([0, 25, 50, 75, 100])
        ax.set_yticks([0, 25, 50, 75, 100])
        ax.grid(color="#eeeeee", linewidth=0.8)
        ax.set_axisbelow(True)

        for point in panel_points:
            ax.scatter(
                numeric(point["model_endorsement"]),
                numeric(point["human_endorsement"]),
                marker=MARKERS.get(point["inference_group"], "o"),
                s=58,
                color=COLORS[point["statement_type"]],
                edgecolor="#333333",
                linewidth=0.45,
                alpha=0.86,
                zorder=3,
            )

        xs = [numeric(point["model_endorsement"]) for point in panel_points]
        ys = [numeric(point["human_endorsement"]) for point in panel_points]
        pairs = [(x, y) for x, y in zip(xs, ys) if x is not None and y is not None]
        xs = [x for x, _ in pairs]
        ys = [y for _, y in pairs]
        regression = linear_regression(xs, ys)
        if regression is not None:
            slope, intercept = regression
            line_x = [0, 100]
            line_y = [intercept + slope * x for x in line_x]
            ax.plot(line_x, line_y, color="#222222", linewidth=1.1, zorder=2)

        pearson, _ = pearson_result(xs, ys)
        pearson_text = f"{pearson:.2f}" if pearson is not None else "NA"
        ax.text(
            0.03,
            0.97,
            f"n = {len(pairs)}\nPearson r = {pearson_text}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=8.5,
            color="#555555",
        )
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)

    axes[0].set_ylabel("Human endorsement", fontsize=11)
    for ax in axes:
        ax.set_xlabel("VLM endorsement", fontsize=11)

    color_handles = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            markerfacecolor=COLORS[statement_type],
            markeredgecolor="#333333",
            markeredgewidth=0.45,
            markersize=7,
            label=STATEMENT_LABELS[statement_type],
        )
        for statement_type in STATEMENT_TYPES
    ]
    marker_handles = [
        plt.Line2D(
            [0],
            [0],
            marker=MARKERS[group],
            linestyle="",
            color="#555555",
            markerfacecolor="#eeeeee",
            markersize=7,
            label=group,
        )
        for group, _ in INFERENCE_ORDER
    ]
    fig.legend(
        handles=color_handles + marker_handles,
        loc="lower center",
        ncol=6,
        frameon=False,
        fontsize=8.5,
        bbox_to_anchor=(0.5, -0.01),
    )
    fig.tight_layout(rect=[0, 0.08, 1, 0.94])

    output_dir.mkdir(parents=True, exist_ok=True)
    for fmt in formats:
        fig.savefig(output_dir / f"{model_slug(raw_model)}_human_scatter.{fmt}", bbox_inches="tight")
    plt.close(fig)


def plot_all(points: list[dict], output_dir: Path, formats: tuple[str, ...]) -> None:
    by_model: dict[str, list[dict]] = defaultdict(list)
    for point in points:
        by_model[point["raw_model"]].append(point)

    for raw_model, model_points in sorted(by_model.items()):
        plot_model(raw_model, model_points, output_dir, formats)
        print(f"Wrote scatter figure for {raw_model}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot human endorsement against VLM endorsement by condition."
    )
    parser.add_argument("--ratings", type=Path, default=DEFAULT_RATINGS)
    parser.add_argument("--human", type=Path, default=DEFAULT_HUMAN)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--formats", nargs="+", default=["png", "pdf"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rating_rows = load_csv(args.ratings)
    human_rows = load_csv(args.human)
    points = sorted(build_points(rating_rows, human_rows), key=sort_key)
    if not points:
        raise RuntimeError("No model/human scatter points could be built.")

    point_path = args.output_dir / "model_human_scatter_points.csv"
    write_csv(
        point_path,
        points,
        [
            "raw_model",
            "input_mode",
            "inference_group",
            "condition",
            "premise",
            "statement_type",
            "model_endorsement",
            "human_endorsement",
            "n_model_videos",
        ],
    )
    agreement_path = args.output_dir / "model_human_raw_condition_agreement.csv"
    write_csv(
        agreement_path,
        agreement_rows(points),
        [
            "model",
            "input_format",
            "n_conditions",
            "mae",
            "pearson_r",
            "pearson_p",
            "spearman_rho",
            "spearman_p",
        ],
    )
    largest_error_path = args.output_dir / "model_human_largest_raw_condition_errors.csv"
    write_csv(
        largest_error_path,
        largest_error_rows(points),
        [
            "model",
            "input_format",
            "rank",
            "inference_group",
            "condition",
            "premise",
            "statement_type",
            "model_endorsement",
            "human_endorsement",
            "absolute_error",
            "n_model_videos",
        ],
    )
    plot_all(points, args.output_dir, tuple(args.formats))
    print(f"Wrote plotted points to {point_path}")
    print(f"Wrote raw-condition agreement metrics to {agreement_path}")
    print(f"Wrote largest raw-condition errors to {largest_error_path}")


if __name__ == "__main__":
    main()
