from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import numpy as np
from scipy import stats


ROOT = Path(__file__).resolve().parents[1]
RAW_POINTS = ROOT / "results" / "model_human_scatter" / "model_human_scatter_points.csv"
EFFECTS = ROOT / "results" / "model_comparison" / "model_human_effects.csv"
RAW_OUT = ROOT / "results" / "human_likeness_regression"
EFFECT_OUT = ROOT / "results" / "effect_human_likeness_regression"


def read_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def architecture(model: str) -> str:
    name = model.lower()
    if "qwen3.8" in name:
        return "Qwen3.8"
    if "qwen3-vl" in name:
        return "Qwen3-VL"
    if "qwen2.5-vl" in name:
        return "Qwen2.5-VL"
    if "gemma-4" in name:
        return "Gemma 4"
    return re.sub(r"[-_/]+", " ", model).strip()


def input_format(mode: str) -> str:
    return {"pure-text": "text", "pure-video": "video", "split": "split"}.get(mode, mode)


def base_row(row: dict, error: float, experimental_condition: str) -> dict:
    return {
        "raw_model": row["raw_model"],
        "architecture": architecture(row["raw_model"]),
        "input_format": input_format(row["input_mode"]),
        "experimental_condition": experimental_condition,
        "inference_group": row["inference_group"],
        "condition": row["condition"],
        "absolute_error": f"{error:.6f}",
        "human_likeness": f"{100 - error:.6f}",
    }


def raw_rows(rows: list[dict]) -> list[dict]:
    out = []
    for row in rows:
        model, human = float(row["model_endorsement"]), float(row["human_endorsement"])
        condition = "/".join([row["inference_group"], row["condition"], row["premise"], row["statement_type"]])
        out.append(
            base_row(row, abs(model - human), condition)
            | {
                "premise": row["premise"],
                "statement_type": row["statement_type"],
                "model_endorsement": f"{model:.6f}",
                "human_endorsement": f"{human:.6f}",
            }
        )
    return out


def effect_rows(rows: list[dict]) -> list[dict]:
    out = []
    for row in rows:
        condition = "/".join([row["inference_group"], row["condition"], row["effect_type"]])
        out.append(
            base_row(row, float(row["absolute_error"]), condition)
            | {
                "effect_type": row["effect_type"],
                "model_effect": row["model_effect"],
                "human_effect": row["human_effect"],
            }
        )
    return out


def indicator(rows: list[dict], column: str, level: str) -> np.ndarray:
    return np.array([row[column] == level for row in rows], dtype=float)


def design_matrix(
    rows: list[dict],
    predictors: list[str],
    interactions: list[tuple[str, str]],
) -> tuple[np.ndarray, list[str], list[str]]:
    columns = [np.ones(len(rows))]
    terms = ["Intercept"]
    references = []
    model_columns = list(dict.fromkeys(predictors + [name for pair in interactions for name in pair]))
    levels = {name: sorted({row[name] for row in rows}) for name in model_columns}

    for name in model_columns:
        references.append(f"{name}: {levels[name][0]}")
        if name in predictors:
            for level in levels[name][1:]:
                terms.append(f"{name}={level}")
                columns.append(indicator(rows, name, level))

    for left, right in interactions:
        for left_level in levels[left][1:]:
            for right_level in levels[right][1:]:
                terms.append(f"{left}={left_level}:{right}={right_level}")
                columns.append(indicator(rows, left, left_level) * indicator(rows, right, right_level))

    return np.column_stack(columns), terms, references


def fit_ols(rows: list[dict], predictors: list[str], interactions: list[tuple[str, str]]) -> tuple[list[dict], dict]:
    y = np.array([float(row["human_likeness"]) for row in rows])
    x, terms, references = design_matrix(rows, predictors, interactions)
    beta, *_ = np.linalg.lstsq(x, y, rcond=None)
    residuals = y - x @ beta
    n, p = x.shape
    df_model, df_resid = p - 1, n - p
    rss = float(np.sum(residuals**2))
    tss = float(np.sum((y - y.mean()) ** 2))
    r2 = 1 - rss / tss
    f_value = (r2 / df_model) / ((1 - r2) / df_resid)
    se = np.sqrt(np.diag((rss / df_resid) * np.linalg.pinv(x.T @ x)))
    t_values = beta / se
    p_values = 2 * stats.t.sf(np.abs(t_values), df_resid)
    coefficients = [
        {
            "term": term,
            "estimate": f"{estimate:.6f}",
            "std_error": f"{std_error:.6f}",
            "t_value": f"{t_value:.6f}",
            "p_value": f"{p_value:.6f}",
        }
        for term, estimate, std_error, t_value, p_value in zip(terms, beta, se, t_values, p_values)
    ]
    return coefficients, {
        "n": n,
        "df_model": df_model,
        "df_resid": df_resid,
        "f_value": f_value,
        "f_p_value": stats.f.sf(f_value, df_model, df_resid),
        "r2": r2,
        "adj_r2": 1 - (1 - r2) * (n - 1) / df_resid,
        "references": references,
        "interactions": interactions,
    }


def write_summary(path: Path, predictors: list[str], summary: dict) -> None:
    interactions = ", ".join(f"{left}:{right}" for left, right in summary["interactions"]) or "none"
    lines = [
        "Linear regression predicting human likeness",
        "Outcome: 100 - absolute_error",
        f"Predictors: {', '.join(predictors)}",
        f"Interactions: {interactions}",
        f"N: {summary['n']}",
        f"F({summary['df_model']}, {summary['df_resid']}) = {summary['f_value']:.6f}",
        f"Model p-value: {summary['f_p_value']:.10g}",
        f"R-squared: {summary['r2']:.6f}",
        f"Adjusted R-squared: {summary['adj_r2']:.6f}",
        "",
        "Reference levels:",
        *[f"- {ref}" for ref in summary["references"]],
        "",
        "Positive coefficients mean more human-like than the reference level.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--level", choices=("raw", "effect"), default="raw")
    parser.add_argument("--points", type=Path, default=RAW_POINTS)
    parser.add_argument("--effects", type=Path, default=EFFECTS)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--no-input-format", action="store_true")
    parser.add_argument("--interactions", action="store_true")
    args = parser.parse_args()

    rows = effect_rows(read_csv(args.effects)) if args.level == "effect" else raw_rows(read_csv(args.points))
    output_dir = args.output_dir or (EFFECT_OUT if args.level == "effect" else RAW_OUT)
    predictors = ["architecture", "experimental_condition"]
    if not args.no_input_format:
        predictors.insert(1, "input_format")
    interactions = [("architecture", "inference_group")] if args.interactions else []
    if args.interactions and not args.no_input_format:
        interactions += [("architecture", "input_format"), ("input_format", "inference_group")]

    coefficients, summary = fit_ols(rows, predictors, interactions)
    prefix = "effect_human_likeness_regression" if args.level == "effect" else "human_likeness_regression"
    if args.interactions:
        prefix += "_with_interactions"
    write_csv(output_dir / f"{prefix}_rows.csv", rows)
    write_csv(output_dir / f"{prefix}_coefficients.csv", coefficients)
    write_summary(output_dir / f"{prefix}_summary.txt", predictors, summary)
    print(f"Wrote {args.level} regression outputs to {output_dir}")


if __name__ == "__main__":
    main()
