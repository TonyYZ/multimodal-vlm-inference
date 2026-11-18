import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
from item_effect_metrics import effect_summary, summarize_effects
from analyze_model_size import model_size_b


class ItemEffectMetricsTest(unittest.TestCase):
    def test_unavailable_test_does_not_remove_an_available_contrast(self):
        frame = pd.DataFrame({"model": ["m"] * 7, "input_mode": ["split"] * 7,
                              "item_contrast_difference": [0] * 6 + [7],
                              "p_value": [np.nan] * 7, "effect_detected": [np.nan] * 7})
        result = summarize_effects(frame).iloc[0]
        self.assertAlmostEqual(result.effect_rms_distance, np.sqrt(7))
        self.assertEqual(result.n_finite_item_contrasts, 7)
        self.assertEqual(result.n_unavailable_wilcoxon, 7)
        frame.loc[0, "item_contrast_difference"] = np.nan
        self.assertTrue(np.isnan(summarize_effects(frame).iloc[0].effect_rms_distance))

    def test_distances_match_existing_contrast_clustering(self):
        for mode in ("pure-text", "pure-video", "split"):
            path = ROOT / "results/animation/inference/effect_profiles/item_effects" / f"mean_item_contrasts_{mode.replace('-', '_')}_distances.csv"
            distances = pd.read_csv(path, index_col="model")
            for row in effect_summary().query("input_mode == @mode").itertuples():
                self.assertAlmostEqual(row.effect_rms_distance, distances.loc[row.model, "Human"])

    def test_total_parameter_sizes(self):
        self.assertEqual(model_size_b("Qwen/Qwen3-Omni-30B-A3B-Instruct"), 30)
        self.assertEqual(model_size_b("openbmb/MiniCPM-o-4_5"), 9)
        self.assertEqual(model_size_b("Qwen/Qwen2.5-VL-32B-Instruct"), 32)


if __name__ == "__main__":
    unittest.main()
