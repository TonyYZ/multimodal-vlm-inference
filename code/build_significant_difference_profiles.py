from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap
from scipy.stats import false_discovery_control, ttest_1samp, ttest_ind

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EFFECTS = ROOT / "results/animation/human_tests/llm_fixed_benchmark_tests.csv"
DEFAULT_OUTPUT = ROOT / "results/animation/inference/effect_profiles/legacy_benchmark"
MODEL_ORDER = (
    "Qwen/Qwen2.5-VL-3B-Instruct",
    "Qwen/Qwen2.5-VL-7B-Instruct",
    "Qwen/Qwen2.5-VL-32B-Instruct",
    "Qwen/Qwen3-VL-2B-Instruct",
    "Qwen/Qwen3-VL-8B-Instruct",
    "Qwen/Qwen3.8-27B",
    "google/gemma-4-12B-it",
)
MODES = ("pure-text", "pure-video", "split")
TEST_ORDER = (
    "Scalar implicature: Positive",
    "Scalar implicature: Negative",
    "Presupposition: Question",
    "Presupposition: None",
    "Supplement: Unembedded",
    "Homogeneity: Positive",
    "Homogeneity: Negative",
)


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def clean(value: object) -> str:
    return "None" if pd.isna(value) or str(value).strip() == "" else str(value).strip()


def short_model(model: str) -> str:
    return model.split("/")[-1].replace("-Instruct", "").replace("-it", "")


def one_sample_result(values: pd.Series, model_effect: float) -> dict[str, float]:
    values = pd.to_numeric(values, errors="coerce").dropna()
    n = len(values)
    if n < 2 or not np.isfinite(model_effect) or values.std(ddof=1) <= 0:
        raise ValueError("A t-test requires a finite model effect and variable human observations.")
    human_effect = float(values.mean())
    human_sd = float(values.std(ddof=1))
    se = human_sd / np.sqrt(n)
    result = ttest_1samp(values, popmean=model_effect)
    signed_t = -float(result.statistic)
    return {
        "human_effect": human_effect,
        "human_sd": human_sd,
        "human_se": se,
        "n_human": n,
        "df": n - 1,
        "signed_t": signed_t,
        "absolute_t": abs(signed_t),
        "p_value": float(result.pvalue),
    }


def independent_difference_result(
    target: pd.Series, control: pd.Series, model_effect: float
) -> dict[str, float]:
    target = pd.to_numeric(target, errors="coerce").dropna()
    control = pd.to_numeric(control, errors="coerce").dropna()
    nt, nc = len(target), len(control)
    if min(nt, nc) < 2 or not np.isfinite(model_effect):
        raise ValueError("Welch's test requires at least two participants per group.")
    vt, vc = target.var(ddof=1), control.var(ddof=1)
    se2_t, se2_c = vt / nt, vc / nc
    se = np.sqrt(se2_t + se2_c)
    if not np.isfinite(se) or se <= 0:
        raise ValueError("The independent contrast has no estimable standard error.")
    # Shift the independent contrast's null to the fixed model effect.
    result = ttest_ind(target - model_effect, control, equal_var=False)
    human_effect = float(target.mean() - control.mean())
    signed_t = -float(result.statistic)
    return {
        "human_effect": human_effect,
        "human_sd": np.nan,
        "human_se": se,
        "n_human": nt + nc,
        "df": float(result.df),
        "signed_t": signed_t,
        "absolute_t": abs(signed_t),
        "p_value": float(result.pvalue),
    }


def animation_tests(effects_path: Path = DEFAULT_EFFECTS, alpha: float = 0.05) -> pd.DataFrame:
    frame = pd.read_csv(effects_path, keep_default_na=False).rename(
        columns={"input_format": "input_mode"}
    )
    for column in ("estimate", "chi_square", "df", "p_value", "human_diagnostic_coefficient", "llm_diagnostic_coefficient"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    valid = (
        frame["analysis_status"].eq("ok") & frame["df"].eq(1)
        & np.isfinite(frame[["estimate", "chi_square", "p_value"]]).all(axis=1)
    )
    if (frame.loc[valid, "chi_square"] < 0).any():
        raise ValueError("Likelihood-ratio chi-square cannot be negative.")
    frame["inference_group"] = frame["condition"].replace({"Scalar Implicature": "Scalar implicature"})
    frame["test"] = frame["inference_group"] + ": " + frame["environment"]
    if frame.duplicated(["model", "input_mode", "test"]).any():
        raise ValueError("Duplicate model/format/diagnostic results.")
    if any(set(group["test"]) != set(TEST_ORDER) for _, group in frame.groupby(["model", "input_mode"])):
        raise ValueError("Each model/format must include all seven diagnostics, with failed fits marked unavailable.")
    frame.loc[~valid, "p_value"] = np.nan
    frame["study"] = "Animation"
    frame["test_design"] = "Participant-level human-minus-fixed-LLM mixed-model LRT"
    frame["bh_family_size"] = len(TEST_ORDER)
    frame["alpha"] = alpha
    return adjust_p_values(frame, alpha)


def benchmark_summary(effects_path: Path = DEFAULT_EFFECTS, alpha: float = 0.05) -> pd.DataFrame:
    """Count detected differences; a complete seven-test profile is required."""
    frame = animation_tests(effects_path, alpha)
    return frame.groupby(["model", "input_mode"], as_index=False).agg(
        n_conditions=("significantly_different", "count"),
        n_differences=("significantly_different", lambda values: values.sum() if values.notna().all() else np.nan),
        fraction_significantly_different=("significantly_different", lambda values: values.mean() if values.notna().all() else np.nan),
    )


def load_emoji_responses(path: Path, response_type: str, response_value: str) -> pd.DataFrame:
    data = pd.read_csv(path, low_memory=False)
    data = data[data[response_type] == response_value].copy()
    data["Response"] = pd.to_numeric(data["Response"], errors="coerce")
    return data.dropna(subset=["Response"])


def paired_emoji_effects(data: pd.DataFrame, group_columns: list[str]) -> dict[tuple, pd.Series]:
    grouped = data.groupby(
        [*group_columns, "Participant Private ID", "human_condition"], dropna=False
    )["Response"].mean().unstack("human_condition")
    output = defaultdict(list)
    for index, row in grouped.dropna(subset=["Target", "Control"]).iterrows():
        group_key = index[:-1] if isinstance(index, tuple) else (index,)
        output[tuple(clean(value) for value in group_key)].append(
            row["Target"] - row["Control"]
        )
    return {key: pd.Series(values) for key, values in output.items()}


def emoji_human_effects() -> tuple[dict[tuple, tuple[pd.Series, pd.Series]], dict[tuple, pd.Series]]:
    source = ROOT / "materials/emoji_study/source_cache"
    exp1 = {}
    for material, filename in (("Target", "exp1_target.csv"), ("Control", "exp1_control.csv")):
        data = load_emoji_responses(
            source / filename, "Zone Type", "response_slider_endValue"
        )
        data["Environment"] = data["Environment"].map(clean)
        for (environment, inference), group in data[
            data["Emoji"] == "Smoke"
        ].groupby(["Environment", "Inference.Type"]):
            exp1[(material, clean(environment), clean(inference))] = group.groupby(
                "Participant Private ID"
            )["Response"].mean()
    independent = {
        ("Experiment 1", "Cigarette", environment, inference): (
            exp1[("Target", environment, inference)],
            exp1[("Control", environment, inference)],
        )
        for material, environment, inference in exp1
        if material == "Target" and ("Control", environment, inference) in exp1
    }

    exp2 = load_emoji_responses(
        source / "exp2_protext.csv", "Zone Type", "response_slider_endValue"
    )
    exp2 = exp2[exp2["Emoji"] == "Plane_takeoff"].copy()
    exp2["human_condition"] = exp2["Condition"]
    exp2["Environment"] = exp2["Environment"].map(clean)
    paired2 = paired_emoji_effects(exp2, ["Environment"])

    exp3 = load_emoji_responses(source / "exp3_task.csv", "Response Type", "response")
    exp3["human_condition"] = exp3["Spreadsheet: Condition"]
    paired3 = paired_emoji_effects(exp3, ["Spreadsheet: Emoji"])
    paired = {
        **{
            ("Experiment 2", "Plane_takeoff", environment, "Target minus control"): values
            for (environment,), values in paired2.items()
        },
        **{
            ("Experiment 3", emoji, "Unembedded", "Target minus control"): values
            for (emoji,), values in paired3.items()
        },
    }
    return independent, paired


def emoji_tests(effects_path: Path, alpha: float = 0.05) -> pd.DataFrame:
    effects = pd.read_csv(effects_path, keep_default_na=False)
    independent, paired = emoji_human_effects()
    rows = []
    for _, row in effects.iterrows():
        key = (
            clean(row["experiment"]),
            clean(row["emoji"]),
            clean(row["environment"]),
            clean(row["inference_type"]),
        )
        model_effect = float(row["model_effect"])
        if key in independent:
            result = independent_difference_result(*independent[key], model_effect)
            design = "Welch t-test for independent human contrast"
        else:
            result = one_sample_result(paired[key], model_effect)
            design = "one-sample t-test on paired participant effects"
        rows.append(
            {
                "study": "Emoji",
                "model": row["model"],
                "input_mode": "pure-text",
                "test": " / ".join(key),
                "model_effect": model_effect,
                **result,
                "test_design": design,
            }
        )
    return adjust_p_values(pd.DataFrame(rows), alpha)


def adjust_p_values(frame: pd.DataFrame, alpha: float = 0.05) -> pd.DataFrame:
    frame["q_value"] = np.nan
    for _, indices in frame.groupby(["model", "input_mode"]).groups.items():
        # Keep the full diagnostic family even when a fit is unavailable.
        frame.loc[indices, "q_value"] = false_discovery_control(
            frame.loc[indices, "p_value"].fillna(1).to_numpy(), method="bh"
        )
    frame.loc[frame["p_value"].isna(), "q_value"] = np.nan
    frame["significantly_different"] = (frame["q_value"] < alpha).astype(float).where(frame["q_value"].notna())
    return frame


def profile_matrix(rows: list[dict], mode: str) -> tuple[list[str], list[str], np.ndarray]:
    models = [model for model in MODEL_ORDER if any(row["model"] == model for row in rows)]
    tests = [test for test in TEST_ORDER if any(row["test"] == test for row in rows)]
    lookup = {
        (row["model"], row["input_mode"], row["test"]): row["significantly_different"]
        for row in rows
    }
    matrix = np.array(
        [[lookup[(model, mode, test)] for test in tests] for model in models],
        dtype=float,
    )
    return models, tests, matrix


def plot_heatmaps(rows: list[dict], output_dir: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(17, 6.5), sharey=True, constrained_layout=True)
    image = None
    for ax, mode in zip(axes, MODES):
        models, tests, matrix = profile_matrix(rows, mode)
        cmap = ListedColormap(["#7BC87C", "#D65F5F"])
        cmap.set_bad("#d9d9d9")
        image = ax.imshow(
            matrix,
            cmap=cmap,
            vmin=0,
            vmax=1,
            aspect="auto",
        )
        ax.set_xticks(range(len(tests)), tests, rotation=43, ha="right", fontsize=8)
        ax.set_yticks(range(len(models)), [short_model(model) for model in models])
        ax.set_title(mode)
        for y in range(len(models)):
            for x in range(len(tests)):
                label = str(int(matrix[y, x])) if np.isfinite(matrix[y, x]) else "NA"
                ax.text(x, y, label, ha="center", va="center", fontsize=8)
    alpha = rows[0]["alpha"]
    fig.suptitle(f"Fixed-benchmark diagnostic LRTs: BH-adjusted q < {alpha:g}\nIndividual human ratings; singular fits retained; gray = unavailable", fontsize=12)
    colorbar = fig.colorbar(image, ax=axes, ticks=[0.25, 0.75], shrink=0.72)
    colorbar.ax.set_yticklabels(["0: no detected difference", "1: significant difference"])
    for suffix in ("png", "pdf"):
        fig.savefig(output_dir / f"significant_difference_heatmaps.{suffix}", dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot item Wilcoxon profiles; --benchmark selects archived human-LLM regression tests."
    )
    parser.add_argument("--effects", type=Path, default=DEFAULT_EFFECTS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--sensitivity", type=Path, help="Summarize a random-effects sensitivity folder only; leave main plots unchanged.")
    parser.add_argument("--fixed-vlm", action="store_true", help="Plot VLM-only fixed-effect performance, not discrepancy tests.")
    parser.add_argument("--benchmark", action="store_true", help="Explicitly rerun the older human-minus-fixed-LLM comparison instead.")
    parser.add_argument("--study", choices=("animation", "gesture"), default="animation")
    args = parser.parse_args()
    if not 0 < args.alpha < 1:
        parser.error("--alpha must be between 0 and 1.")
    if args.fixed_vlm and args.benchmark:
        parser.error("Choose --fixed-vlm or --benchmark, not both.")
    if not args.benchmark and not args.sensitivity:
        from plot_item_effects import regenerate
        regenerate(args.study, args.alpha)
        return
    if args.sensitivity:
        from build_binary_effect_profiles import render_sensitivity
        render_sensitivity(args.sensitivity, args.alpha)
        return
    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame = animation_tests(args.effects, args.alpha)
    frame.assign(significantly_different=frame["significantly_different"].astype("Int64")).to_csv(
        args.output_dir / "model_human_significance_tests.csv", index=False, na_rep="NA"
    )
    rows = frame.to_dict("records")
    plot_heatmaps(rows, args.output_dir)
    from build_binary_effect_profiles import render_profiles
    render_profiles(frame, args.output_dir)
    benchmark_summary(args.effects, args.alpha).to_csv(args.output_dir / "model_human_benchmark_summary.csv", index=False)
    print(f"Wrote BH-adjusted binary difference profiles to {args.output_dir}")


if __name__ == "__main__":
    main()
