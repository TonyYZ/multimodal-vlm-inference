"""Compare temperature-sampled model ratings with individual human ratings (animation).

Each sample index is treated as one pseudo-participant. Three levels are tested:
  cells    - per-cell rating distributions (Mann-Whitney U and KS), BH-FDR across cells
  effects  - per-participant test contrasts d (as in item_effect_tests.py, averaged over stories),
             model pseudo-participants vs humans (Mann-Whitney U, mean difference + bootstrap CI),
             plus a TOST equivalence test: is d_model - d_human reliably within +/- --margin points?
  profile  - typicality: distance of each participant's full cell profile from the
             leave-one-out human mean, model pseudo-participants vs the human distribution
"""
import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import false_discovery_control, ks_2samp, mannwhitneyu, t, ttest_ind, wasserstein_distance

ROOT = Path(__file__).resolve().parents[1]
HUMAN = ROOT / "materials" / "analyses"
CELLS = ROOT / "results" / "animation" / "human_tests" / "llm_fixed_effect_cells.csv"
OUT = ROOT / "results" / "animation" / "sampling_pilot"
KEY = ["condition", "environment", "Story", "GesturalPremise", "InferenceType"]
TEST = ["condition", "environment"]


def parse_score(response):
    """Leading number only, as in vlm_fixed_effects.R (a leaked Gemma 'thought' line is skipped)."""
    text = re.sub(r"^\s*thought[ \t]*\r?\n", "", str(response), flags=re.IGNORECASE)
    match = re.match(r"\s*(\d{1,3}(?:\.\d+)?)(?:\s|$)", text)
    return float(match.group(1)) if match and float(match.group(1)) <= 100 else None


def human_ratings():
    data = pd.read_csv(HUMAN / "TSC_EXP_2_Data.csv", dtype=str).iloc[2:]
    cols = [c for c in data.columns if c.endswith("_Test_2")]
    long = data.melt(id_vars=["ResponseId"], value_vars=cols, var_name="qualtrics", value_name="value")
    long = long[long.value.notna() & long.value.str.strip().ne("")]
    long["qualtrics"] = long.qualtrics.str.removesuffix("_Test_2")
    design = pd.read_csv(HUMAN / "TSC_EXP_2_Design.csv", keep_default_na=False)
    long = long.merge(design, on="qualtrics", validate="many_to_one")
    long["condition"] = long.Condition.replace({"Implicature": "Scalar Implicature"})
    long["environment"] = long.Environment
    long["InferenceType"] = long.InferenceType.replace({"Control": "Baseline"})
    long["value"] = pd.to_numeric(long.value)
    return long.rename(columns={"ResponseId": "participant"})[["participant", *KEY, "value"]]


def model_ratings(path):
    cells = pd.read_csv(CELLS, keep_default_na=False).drop_duplicates(["source_video", "statement_type"])
    rows = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    frame = pd.DataFrame(rows)
    frame["source_video"] = frame.source_video.str.replace("\\", "/").str.split("/").str[-1]
    frame["value"] = frame.response.map(parse_score)
    frame = frame.merge(cells[["source_video", "statement_type", *KEY]], on=["source_video", "statement_type"],
                        validate="many_to_one")
    frame["participant"] = "sample_" + frame["sample"].astype(str)
    return frame


def contrasts(ratings):
    """Per participant and test: story-averaged contrast d (DID for SI/Supplement)."""
    wide = ratings.pivot_table(index=["participant", *TEST, "Story"], columns=["GesturalPremise", "InferenceType"],
                               values="value")
    d = wide[("Target", "Target")] - wide[("Target", "Baseline")]
    did = wide.index.get_level_values("condition").isin(["Scalar Implicature", "Supplement"])
    d = d.where(~did, d - (wide.get(("Control", "Target")) - wide.get(("Control", "Baseline"))))
    return d.groupby(["participant", *TEST]).mean().rename("d").reset_index()


def literal_check(ratings):
    """Per participant: literally true minus literally false statement, SI positive control premise."""
    cell = ratings[(ratings.condition == "Scalar Implicature") & (ratings.environment == "Positive")
                   & (ratings.GesturalPremise == "Control")]
    wide = cell.groupby(["participant", "InferenceType"]).value.mean().unstack()
    return wide.Baseline - wide.Target


def bootstrap_diff(a, b, rng, n=5000):
    draws = [rng.choice(a, len(a)).mean() - rng.choice(b, len(b)).mean() for _ in range(n)]
    return np.percentile(draws, [2.5, 97.5])


def equivalence_test(a, b, margin):
    """TOST (Welch): H0 |mean(a) - mean(b)| >= margin. Equivalent at .05 iff the 90% CI lies within +/- margin."""
    p_lower = ttest_ind(a + margin, b, equal_var=False, alternative="greater").pvalue
    p_upper = ttest_ind(a - margin, b, equal_var=False, alternative="less").pvalue
    va, vb = a.var(ddof=1) / len(a), b.var(ddof=1) / len(b)
    df = (va + vb) ** 2 / (va ** 2 / (len(a) - 1) + vb ** 2 / (len(b) - 1))
    half = t.ppf(.95, df) * np.sqrt(va + vb)
    diff = a.mean() - b.mean()
    return dict(tost_p=max(p_lower, p_upper), ci90_low=diff - half, ci90_high=diff + half)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("ratings", nargs="+", type=Path)
    parser.add_argument("--margin", type=float, default=20,
                        help="Equivalence margin for d_model - d_human, in rating points (default 20).")
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    human = human_ratings()
    human_d = contrasts(human)
    profile = human.pivot_table(index="participant", columns=KEY, values="value")
    cell_rows, effect_rows, profile_rows, valid_rows = [], [], [], []
    pooled = [human.assign(source="Human")[["source", "participant", *KEY, "value"]]]
    human_check_p10 = literal_check(human).quantile(.10)
    # Leave-one-out human typicality: RMS distance of each person's profile from the mean of the others.
    total, n_h = profile.sum(), len(profile)
    human_dist = np.sqrt((((profile - (total - profile) / (n_h - 1)) ** 2).mean(axis=1)))
    for path in args.ratings:
        model = model_ratings(path)
        name = f"{model.model.iloc[0]} | {model.input_mode.iloc[0]} | T={model.temperature.iloc[0]}"
        valid_rows.append(dict(model=name, n_responses=len(model), parse_rate=model.value.notna().mean(),
                               n_samples=model["sample"].nunique()))
        # Inclusion: usable ratings >= 95% and the median pseudo-participant's literal check above the
        # human 10th percentile (i.e. at least half of the pseudo-participants pass the human cutoff).
        check = literal_check(model[model.value.notna()])
        valid_rows[-1].update(literal_check_mean=check.mean(), literal_check_median=check.median(),
                              human_literal_check_p10=human_check_p10,
                              included=(valid_rows[-1]["parse_rate"] >= .95) and check.median() > human_check_p10)
        model = model[model.value.notna()]
        pooled.append(model.assign(source=f"{model.model.iloc[0]} | {model.input_mode.iloc[0]}")[pooled[0].columns])
        for key, m in model.groupby(KEY):
            h = human.set_index(KEY).loc[key, "value"]
            cell_rows.append(dict(zip(KEY, key), model=name, human_n=len(h), model_n=len(m),
                                  human_mean=h.mean(), model_mean=m.value.mean(), human_sd=h.std(),
                                  model_sd=m.value.std(), wasserstein=wasserstein_distance(m.value, h),
                                  mwu_p=mannwhitneyu(m.value, h).pvalue, ks_p=ks_2samp(m.value, h).pvalue))
        model_d = contrasts(model)
        for key, h in human_d.groupby(TEST):
            m = model_d.set_index(TEST).loc[key, "d"].dropna().to_numpy()
            low, high = bootstrap_diff(m, h.d.to_numpy(), rng)
            effect_rows.append(dict(zip(TEST, key), model=name, human_n=len(h), model_n=len(m),
                                    human_mean_d=h.d.mean(), model_mean_d=m.mean(), difference=m.mean() - h.d.mean(),
                                    ci_low=low, ci_high=high, human_sd_d=h.d.std(), model_sd_d=m.std(ddof=1),
                                    mwu_p=mannwhitneyu(m, h.d).pvalue, ks_p=ks_2samp(m, h.d).pvalue,
                                    margin=args.margin, **equivalence_test(m, h.d.to_numpy(), args.margin)))
            effect_rows[-1]["equivalent"] = effect_rows[-1]["tost_p"] < .05
        sample_profile = model.pivot_table(index="participant", columns=KEY, values="value")[profile.columns]
        dist = np.sqrt(((sample_profile - profile.mean()) ** 2).mean(axis=1)).dropna()
        cutoff = np.percentile(human_dist, 95)
        mean_profile = sample_profile.mean()
        profile_rows.append(dict(model=name, n_pseudo_participants=len(dist), human_median_distance=human_dist.median(),
                                 human_95th_distance=cutoff, model_median_distance=dist.median(),
                                 share_within_human_95=(dist <= cutoff).mean(),
                                 mean_profile_distance=np.sqrt(((mean_profile - profile.mean()) ** 2).mean()),
                                 mean_profile_percentile=(human_dist < np.sqrt(((mean_profile - profile.mean()) ** 2).mean())).mean(),
                                 mwu_p=mannwhitneyu(dist, human_dist).pvalue))
    pd.concat(pooled).to_csv(out / "pseudo_participant_ratings.csv", index=False)
    cells = pd.DataFrame(cell_rows)
    for col in ("mwu_p", "ks_p"):
        cells[col.replace("_p", "_q")] = cells.groupby("model")[col].transform(lambda p: false_discovery_control(p))
    effects = pd.DataFrame(effect_rows)
    effects["mwu_q"] = effects.groupby("model").mwu_p.transform(lambda p: false_discovery_control(p))
    for frame, name in ((pd.DataFrame(valid_rows), "validity"), (cells, "cell_distributions"),
                        (effects, "effect_distributions"), (pd.DataFrame(profile_rows), "profile_typicality")):
        frame.to_csv(out / f"{name}.csv", index=False)
    pd.set_option("display.width", 220, "display.max_columns", 30)
    print(pd.DataFrame(valid_rows))
    print(cells.groupby("model")[["wasserstein", "human_sd", "model_sd"]].mean()
          .join(cells.groupby("model").agg(cells_differ_mwu=("mwu_q", lambda q: (q < .05).sum()),
                                           cells_differ_ks=("ks_q", lambda q: (q < .05).sum()), n_cells=("ks_q", "size"))))
    print(effects[["model", *TEST, "human_mean_d", "model_mean_d", "difference", "ci_low", "ci_high",
                   "human_sd_d", "model_sd_d", "mwu_q", "ci90_low", "ci90_high", "tost_p", "equivalent"]]
          .round(3).to_string(index=False))
    print(effects.groupby("model").equivalent.sum().rename(f"tests_equivalent_within_{args.margin:g}"))
    print(pd.DataFrame(profile_rows).round(3).T)


if __name__ == "__main__":
    main()
