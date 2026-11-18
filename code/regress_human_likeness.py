from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm

from analyze_model_size import model_size_b
from item_effect_metrics import DEFAULT_EFFECTS, effect_rows as current_effect_rows


ROOT = Path(__file__).resolve().parents[1]
RAW_POINTS = ROOT / "results" / "animation" / "inference" / "scatter" / "model_human_scatter_points.csv"
EFFECTS = DEFAULT_EFFECTS
RAW_OUT = ROOT / "results" / "animation" / "inference" / "regression" / "raw"
EFFECT_OUT = ROOT / "results" / "animation" / "inference" / "regression" / "effect"
STUDY_EFFECTS = {
    "animation": EFFECTS,
    "gesture": ROOT / "results" / "gesture" / "inference" / "effect_profiles" / "item_effects" / "item_wilcoxon_tests.csv",
    "emoji": ROOT / "results" / "emoji" / "inference" / "effect_profiles" / "item_wilcoxon_tests.csv",
}


def study_effect_rows(study: str) -> pd.DataFrame:
    """Gesture/emoji mean item contrasts minus the human contrast for the same test.

    Contrasts without a human value (gesture Supplement) or a model value are dropped.
    """
    frame = pd.read_csv(STUDY_EFFECTS[study], keep_default_na=False).rename(columns={"input_format": "input_mode"})
    frame["estimate"] = pd.to_numeric(frame.estimate, errors="coerce")
    human = frame[frame.model.eq("Human")][["test", "estimate"]].rename(columns={"estimate": "human_estimate"})
    models = frame[~frame.model.eq("Human")].merge(human, on="test", validate="many_to_one")
    models["inference_group"] = models.condition.replace({"Scalar Implicature": "Scalar implicature"})
    models["environment"] = models.test.str.split(": ", n=1).str[1]  # unique per test (emoji has several per environment)
    models["item_contrast_difference"] = models.estimate - models.human_estimate
    return models[np.isfinite(models.item_contrast_difference)]


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
    if "qwen2.5-omni" in name:
        return "Qwen2.5-Omni"
    if "qwen3-omni" in name:
        return "Qwen3-Omni"
    if "minicpm" in name:
        return "MiniCPM-o"
    return re.sub(r"[-_/]+", " ", model).strip()


def model_size(model: str) -> str:
    name = model.lower()
    match = re.search(r"(\d+(?:\.\d+)?)b", name)
    return f"{match.group(1)}B" if match else "unknown"


def input_format(mode: str) -> str:
    return {"pure-text": "text", "pure-video": "video", "split": "split"}.get(mode, mode)


def base_row(row: dict, error: float, experimental_condition: str) -> dict:
    return {
        "raw_model": row["raw_model"],
        "architecture": architecture(row["raw_model"]),
        "model_size": model_size(row["raw_model"]),
        "log2_size_b": np.log2(model_size_b(row["raw_model"])),
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
        if name == "log2_size_b":
            columns.append(np.array([row[name] for row in rows], dtype=float))
            terms.append(name)
            continue
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
    if np.linalg.matrix_rank(x) != x.shape[1]:
        raise ValueError("Regression predictors are confounded (rank-deficient design).")
    fit = sm.OLS(y, x).fit()
    coefficients = [
        {
            "term": term,
            "estimate": f"{estimate:.6f}",
            "std_error": f"{std_error:.6f}",
            "t_value": f"{t_value:.6f}",
            "p_value": f"{p_value:.6f}",
        }
        for term, estimate, std_error, t_value, p_value in zip(
            terms, fit.params, fit.bse, fit.tvalues, fit.pvalues)
    ]
    adjusted = stats.false_discovery_control(fit.pvalues[1:], method="bh")
    for row, q in zip(coefficients[1:], adjusted):
        row["q_value"] = f"{q:.6f}"
    coefficients[0]["q_value"] = ""
    return coefficients, {
        "n": int(fit.nobs),
        "df_model": int(fit.df_model),
        "df_resid": int(fit.df_resid),
        "f_value": fit.fvalue,
        "f_p_value": fit.f_pvalue,
        "r2": fit.rsquared,
        "adj_r2": fit.rsquared_adj,
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
        "Size uses log2 total billions of parameters; its coefficient is per doubling.",
        "Exploratory OLS: condition/format rows share models; checkpoints are related.",
        "Conventional p-values do not establish independent model-level replication.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--study", choices=tuple(STUDY_EFFECTS), default="animation")
    parser.add_argument("--level", choices=("raw", "effect"), default="raw")
    parser.add_argument("--points", type=Path, default=RAW_POINTS)
    parser.add_argument("--effects", type=Path, default=EFFECTS)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--no-input-format", action="store_true")
    parser.add_argument("--no-model-size", action="store_true")
    parser.add_argument("--interactions", action="store_true")
    args = parser.parse_args()

    if args.study != "animation" and args.level == "raw":
        parser.error("Gesture and emoji use the effect level (mean item contrasts) only.")
    if args.level == "effect":
        effects = study_effect_rows(args.study) if args.study != "animation" else current_effect_rows(args.effects)
        source = effects.rename(columns={
            "model": "raw_model", "environment": "condition", "condition": "original_family"})
        source["effect_type"] = np.where(source.inference_group.isin(
            ["Scalar implicature", "Supplement"]), "target_minus_control_delta", "target_delta")
        if args.study == "emoji":
            source["effect_type"] = "mean_item_contrast"
        source["absolute_error"] = source.item_contrast_difference.abs()
        source["model_effect"], source["human_effect"] = source.estimate, source.human_estimate
        rows = effect_rows(source.to_dict("records"))
    else:
        rows = raw_rows(read_csv(args.points))
    output_dir = args.output_dir or (EFFECT_OUT if args.level == "effect" else RAW_OUT)
    if args.study != "animation" and args.output_dir is None:
        output_dir = ROOT / "results" / args.study / "inference" / "regression" / "effect"
    predictors = ["architecture", "experimental_condition"]
    if not args.no_model_size:
        predictors.insert(1, "log2_size_b")
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
