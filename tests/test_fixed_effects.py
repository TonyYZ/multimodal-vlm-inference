import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2, false_discovery_control

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))
from plot_item_effects import contrast_tree, corrected, human_diagnostics


class FixedEffectProfilesTest(unittest.TestCase):
    def test_contrast_tree_uses_rms_distances(self):
        frame = pd.DataFrame([[0, 0], [3, 4], [6, 8]], index=["a", "b", "c"])
        tree, distances, complete = contrast_tree(frame)
        np.testing.assert_allclose(distances, np.array([[0, 5, 10], [5, 0, 5], [10, 5, 0]]) / np.sqrt(2))
        self.assertTrue(complete.all())
        self.assertEqual(tree.shape, (2, 4))

    def test_contrast_tree_does_not_impute_missing_values(self):
        frame = pd.DataFrame([[0, np.nan], [3, 4], [6, 8]])
        _, distances, complete = contrast_tree(frame)
        self.assertEqual(complete.tolist(), [True, False])
        self.assertEqual(distances[0, 1], 3)
        with self.assertRaises(ValueError):
            contrast_tree(pd.DataFrame([[np.nan], [np.nan]]))

    def test_missing_test_stays_missing_and_counts_in_bh_family(self):
        p = [.001, .02, .1, .4, .8, 1, np.nan]
        frame = pd.DataFrame({"model": ["m"] * 7, "input_format": ["split"] * 7, "p_value": p})
        result = corrected(frame, .05)
        expected = false_discovery_control(np.nan_to_num(p, nan=1))
        np.testing.assert_allclose(result.q_value.iloc[:6], expected[:6])
        self.assertTrue(np.isnan(result.effect_detected.iloc[6]))

    def test_correction_is_separate_for_each_model_and_format(self):
        a = pd.DataFrame({"model": ["a"] * 7, "input_format": ["split"] * 7, "p_value": [.01] * 7})
        b = a.assign(model="b", p_value=.8)
        c = a.assign(input_format="pure-text", p_value=.9)
        result = corrected(pd.concat([a, b, c], ignore_index=True), .05)
        self.assertEqual(result.effect_detected.tolist(), [1.] * 7 + [0.] * 14)

    def test_both_studies_export_seven_human_diagnostics(self):
        root = Path(__file__).resolve().parents[1]
        for study in ("animation", "gesture"):
            human = human_diagnostics(root / "results" / study / "human_tests/original_human_likelihood_ratio_tests.csv")
            self.assertEqual(len(human), 7)
            self.assertFalse(human.duplicated(["condition", "environment"]).any())

    def test_exported_vlm_tests_are_ml_likelihood_ratio_tests(self):
        root = Path(__file__).resolve().parents[1]
        for study in ("animation", "gesture"):
            frame = pd.read_csv(root / "results" / study / "human_tests/llm_fixed_effect_tests.csv")
            valid = frame[frame.analysis_status.eq("ok")]
            expected = 2 * (valid.log_likelihood_full - valid.log_likelihood_reduced)
            np.testing.assert_allclose(valid.chi_square, expected, atol=1e-9)
            np.testing.assert_allclose(valid.p_value, chi2.sf(valid.chi_square, valid.df), atol=1e-12)
            self.assertTrue(valid.df.eq(1).all())
            self.assertTrue(frame.loc[frame.analysis_status.ne("ok"), "p_value"].isna().all())


if __name__ == "__main__":
    unittest.main()
