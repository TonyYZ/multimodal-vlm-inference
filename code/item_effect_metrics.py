"""Descriptive human alignment of mean item-contrast vectors, not significance tests."""
from pathlib import Path

import numpy as np
import pandas as pd

from build_significant_difference_profiles import TEST_ORDER

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EFFECTS = ROOT / "results/animation/inference/effect_profiles/item_effects/item_wilcoxon_tests.csv"


def effect_rows(path: Path = DEFAULT_EFFECTS) -> pd.DataFrame:
    frame = pd.read_csv(path, keep_default_na=False).rename(columns={"input_format": "input_mode"})
    for column in ("estimate", "p_value", "q_value", "effect_detected"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    if frame.duplicated(["model", "input_mode", "test"]).any():
        raise ValueError("Duplicate model/format/inference-type/environment estimates.")
    human = frame[frame.model.eq("Human")][["test", "estimate"]].rename(columns={"estimate": "human_estimate"})
    if len(human) != len(TEST_ORDER) or set(human.test) != set(TEST_ORDER):
        raise ValueError("Expected seven human reference item contrasts.")
    models = frame[~frame.model.eq("Human")].merge(human, on="test", validate="many_to_one")
    if any(set(group.test) != set(TEST_ORDER) for _, group in models.groupby(["model", "input_mode"])):
        raise ValueError("Each model/format must retain all seven effects, including unavailable fits.")
    models["inference_group"] = models.condition.replace({"Scalar Implicature": "Scalar implicature"})
    models["item_contrast_difference"] = models.estimate - models.human_estimate
    return models


def summarize_effects(frame: pd.DataFrame, *, by_family: bool = False) -> pd.DataFrame:
    keys = ["model", "input_mode"] + (["inference_group"] if by_family else [])
    output = []
    for key, group in frame.groupby(keys, sort=True):
        differences = group.item_contrast_difference.to_numpy()
        complete = np.isfinite(differences).all()
        output.append(dict(zip(keys, key)) | {
            "n_conditions": len(group),
            "n_finite_item_contrasts": int(np.isfinite(differences).sum()),
            "effect_rms_distance": float(np.sqrt(np.mean(differences**2))) if complete else np.nan,
            "n_estimable_wilcoxon": int(group.p_value.notna().sum()),
            "n_detected_effects": int(group.effect_detected.sum()),
            "n_unavailable_wilcoxon": int(group.p_value.isna().sum()),
        })
    return pd.DataFrame(output)


def effect_summary(path: Path = DEFAULT_EFFECTS) -> pd.DataFrame:
    return summarize_effects(effect_rows(path))
