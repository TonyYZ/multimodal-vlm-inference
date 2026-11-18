"""Item-level contrasts and exact Wilcoxon tests shared by the three studies."""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import PermutationMethod, wilcoxon

ROOT = Path(__file__).resolve().parents[1]
KEYS = ["model", "input_format", "condition", "environment", "item"]


def signed_rank(values, expected=None):
    # Round subtraction noise before ranking ties; never turn missing cells into zero.
    values = np.round(np.asarray(values, dtype=float), 10)
    finite = values[np.isfinite(values)]
    expected = len(values) if expected is None else expected
    nonzero = int(np.count_nonzero(finite))
    result = dict(estimate=np.nan, median_difference=np.nan, wilcoxon_w=np.nan,
                  p_value=np.nan, n_items=len(finite), n_items_expected=expected,
                  n_nonzero=nonzero, minimum_attainable_p=min(1., 2. ** (1 - nonzero)),
                  analysis_status="incomplete item cells", test_design="Exact two-sided item Wilcoxon signed-rank")
    if not len(finite) or len(finite) != expected:
        return result
    result.update(estimate=float(np.mean(finite)), median_difference=float(np.median(finite)))
    if nonzero == 0:
        result.update(wilcoxon_w=0., p_value=1., analysis_status="all item contrasts zero")
    else:
        test = wilcoxon(finite, zero_method="wilcox", alternative="two-sided",
                        method=PermutationMethod(n_resamples=np.inf))
        result.update(wilcoxon_w=float(test.statistic), p_value=float(test.pvalue), analysis_status="ok")
    return result


def summarize_items(items):
    rows = []
    for key, group in items.groupby(["model", "input_format", "condition", "environment", "test"], sort=False):
        rows.append(dict(zip(["model", "input_format", "condition", "environment", "test"], key))
                    | signed_rank(group.difference))
    return pd.DataFrame(rows)


def _value(table, premise, inference):
    return table.get((premise, inference), np.nan)


def animation_gesture(study):
    output = ROOT / "results" / study / "human_tests"
    cells = pd.read_csv(output / "llm_fixed_effect_cells.csv", keep_default_na=False).rename(
        columns={"Story": "item", "GesturalPremise": "premise", "InferenceType": "inference", "value": "score"})
    for col in ("score", "human_mean"):
        cells[col] = pd.to_numeric(cells[col], errors="coerce")
    human = cells.drop_duplicates(["condition", "environment", "item", "premise", "inference"]).copy()
    human["model"], human["input_format"], human["score"] = "Human", "human", human.human_mean
    cells = pd.concat([cells, human], ignore_index=True)
    cells["prompt_variant"] = np.where(cells.model.eq("Human"), "original human materials", "standard inference rating")
    columns = ["model", "input_format", "condition", "environment", "item", "premise", "inference",
               "score", "prompt_variant", "source_video", "human_mean"]
    cells[columns].to_csv(output / "per_item_scores.csv", index=False, na_rep="NA")
    items = []
    for key, group in cells.groupby(KEYS, sort=False):
        table = group.set_index(["premise", "inference"]).score
        interaction = key[2] in ("Scalar Implicature", "Supplement")
        difference = _value(table, "Target", "Target") - _value(table, "Target", "Baseline")
        if interaction:
            difference -= _value(table, "Control", "Target") - _value(table, "Control", "Baseline")
        row = dict(zip(KEYS, key))
        row.update(test=f"{key[2].replace('Scalar Implicature', 'Scalar implicature')}: {key[3]}",
                   difference=difference, contrast="premise x inference DID" if interaction else "target minus baseline")
        items.append(row)
    items = pd.DataFrame(items)
    items.to_csv(output / "per_item_contrasts.csv", index=False, na_rep="NA")
    if study == "animation":
        cells.loc[cells.model.eq("google/gemma-4-12B-it") & cells.input_format.eq("split"), columns].to_csv(
            output / "yair_gemma_split_per_item_scores.csv", index=False)
    return summarize_items(items)


def emoji(specifications):
    base = ROOT / "results/emoji"
    manifest = pd.read_csv(ROOT / "materials/emoji_study/emoji_pilot_manifest.csv", keep_default_na=False)
    ratings = pd.read_csv(base / "inference/comparison/emoji_pilot_model_human.csv", keep_default_na=False)
    models = list(ratings.model.unique())
    lookup = ratings.set_index(["model", "source_video", "statement_type"]).model_score
    rows = []
    for model in ["Human", *models]:
        for row in manifest.to_dict("records"):
            co = row["experiment"] == "Experiment 1"
            rows.append(dict(model=model, input_format="human" if model == "Human" else "pure-text",
                             condition=row["emoji_type"], experiment=row["experiment"], environment=row["environment"],
                             item=row["emoji"], premise=row["material_condition"] if co else "Target",
                             inference=row["inference_type"] if co else row["statement_type"].title(),
                             score=float(row["human_mean"]) if model == "Human" else
                             lookup.get((model, row["source_video"], row["statement_type"]), np.nan),
                             source_video=row["source_video"], question=row["inference"],
                             prompt_variant="original human materials" if model == "Human" else "standard emoji rating"))
    cells = pd.DataFrame(rows)
    cells.to_csv(base / "human_tests/per_item_scores.csv", index=False, na_rep="NA")
    items = []
    specs = specifications.drop_duplicates("test")
    for spec in specs.to_dict("records"):
        data = cells[cells.experiment.eq(spec["experiment"])]
        if spec["environment"] not in ("All environments", "All emojis"):
            data = data[data.environment.eq(spec["environment"])]
        for key, group in data.groupby(["model", "input_format", "item"], sort=False):
            if spec["experiment"] == "Experiment 1":
                table = group.set_index(["premise", "inference"]).score
                if spec["reading"] == "Interaction":
                    difference = (_value(table, "Target", "Existential") - _value(table, "Target", "Universal")
                                  - _value(table, "Control", "Existential") + _value(table, "Control", "Universal"))
                else:
                    difference = _value(table, "Target", spec["reading"]) - _value(table, "Control", spec["reading"])
            else:
                # Pooled pro-text keeps one observation per emoji, not per environment.
                paired = group.pivot(index="environment", columns="inference", values="score")
                delta = paired.Target - paired.Baseline
                difference = delta.mean() if delta.notna().all() else np.nan
            items.append(dict(model=key[0], input_format=key[1], item=key[2],
                              condition=group.condition.iloc[0], environment=spec["environment"],
                              test=spec["test"], difference=difference))
    items = pd.DataFrame(items)
    items.to_csv(base / "human_tests/per_item_contrasts.csv", index=False, na_rep="NA")
    return summarize_items(items)


def compare_formats():
    """Exploratory paired-model comparisons; positive differences favor split."""
    from item_effect_metrics import effect_summary
    from scipy.stats import false_discovery_control

    base = ROOT / "results/animation"
    effects = effect_summary().rename(columns={"input_mode": "input_format"})
    raw = pd.read_csv(base / "inference/scatter/model_human_raw_condition_agreement.csv")
    perception = pd.read_csv(base / "perception/charts/description_judge_model_summary.csv").rename(
        columns={"input_mode": "input_format"})
    sources = [(effects, "effect_rms_distance", -1), (raw, "mae", -1),
               (raw, "pearson_r", 1), (raw, "spearman_rho", 1),
               (perception[perception.n.eq(24)], "mean_visual_score", 1)]
    rows = []
    for data, metric, direction in sources:
        data = data.copy()
        data["input_format"] = data.input_format.replace({"text": "pure-text", "video": "pure-video"})
        pairs = data[data.input_format.isin(["split", "pure-video"])].pivot(
            index="model", columns="input_format", values=metric).dropna()
        improvement = direction * (pairs.split - pairs["pure-video"])
        test = signed_rank(improvement)
        rows.append(dict(metric=metric, n_models=len(pairs), mean_split=pairs.split.mean(),
                         mean_video=pairs["pure-video"].mean(), n_split_better=int((improvement > 1e-10).sum()),
                         mean_split_advantage=improvement.mean(), wilcoxon_w=test["wilcoxon_w"],
                         p_value=test["p_value"], unit="paired model; related checkpoints are not independent replications"))
    result = pd.DataFrame(rows)
    result["q_value"] = false_discovery_control(result.p_value.fillna(1), method="bh")
    result.to_csv(base / "analysis/inference_perception/input_format_comparison.csv", index=False)
    return result
