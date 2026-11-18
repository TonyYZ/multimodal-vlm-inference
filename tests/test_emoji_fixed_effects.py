import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
from build_emoji_effect_profiles import CONDITION_ORDER, matrices
from plot_item_effects import corrected


class EmojiFixedEffectsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.frame = pd.read_csv(ROOT / "results/emoji/human_tests/emoji_fixed_effect_tests.csv",
                               keep_default_na=False, na_values=["NA"])

    def test_original_test_set_and_pilot_coverage(self):
        frame = self.frame
        self.assertEqual(len(frame), frame.model.nunique() * 19)
        self.assertEqual(set(frame.test), set(CONDITION_ORDER))
        self.assertEqual(frame.groupby("model").size().tolist(), [19] * frame.model.nunique())
        vlm = frame[frame.model.ne("Human")]
        self.assertTrue(vlm.n_participants.eq(0).all())
        ratings = pd.read_csv(ROOT / "results/emoji/inference/comparison/emoji_pilot_model_human.csv", keep_default_na=False)
        for row in vlm.itertuples():
            items = ratings[ratings.model.eq(row.model) & ratings.experiment.eq(row.experiment)].emoji.nunique()
            self.assertEqual(row.n_items, items)

    def test_lrt_matches_log_likelihoods_and_chi_square_tail(self):
        valid = self.frame[self.frame.p_value.notna()]
        np.testing.assert_allclose(valid.chi_square,
                                   2 * (valid.log_likelihood_full - valid.log_likelihood_reduced), atol=1e-9)
        np.testing.assert_allclose(valid.p_value, chi2.sf(valid.chi_square, valid.df), atol=1e-12)
        self.assertTrue(valid.df.eq(1).all())
        unavailable = self.frame[self.frame.analysis_status.ne("ok")]
        self.assertTrue(unavailable.p_value.isna().all())
        withheld = unavailable.analysis_status.str.startswith("incomplete")
        self.assertTrue(unavailable.loc[withheld, "estimate"].isna().all())
        self.assertTrue(np.isfinite(unavailable.loc[~withheld, "estimate"]).all())

    def test_vlm_coefficients_match_original_contrast_coding(self):
        ratings = pd.read_csv(ROOT / "results/emoji/inference/comparison/emoji_pilot_model_human.csv",
                              keep_default_na=False)
        for row in self.frame[self.frame.model.ne("Human")].itertuples():
            if row.analysis_status.startswith("incomplete"):
                self.assertTrue(np.isnan(row.estimate))
                self.assertEqual(row.missing_cells, 1)
                continue
            data = ratings[ratings.model.eq(row.model) & ratings.experiment.eq(row.experiment)]
            if row.environment not in ("All environments", "All emojis"):
                data = data[data.environment.eq(row.environment)]
            if row.experiment == "Experiment 1":
                def contrast(reading):
                    means = data[data.inference_type.eq(reading)].groupby("material_condition").model_score.mean()
                    return means["Control"] - means["Target"]
                expected = (contrast("Universal") - contrast("Existential")
                            if row.reading == "Interaction" else contrast(row.reading))
            else:
                means = data.groupby("statement_type").model_score.mean()
                expected = means["baseline"] - means["target"]
            self.assertAlmostEqual(row.estimate, expected, places=8)

    def test_vectors_preserve_coefficients_with_unavailable_tests(self):
        frame = pd.read_csv(ROOT / "results/emoji/inference/effect_profiles/item_wilcoxon_tests.csv")
        models, tests, coefficients, detected = matrices(frame.to_dict("records"))
        self.assertEqual(models[0], "Human")
        self.assertEqual(tests, list(CONDITION_ORDER))
        self.assertEqual(coefficients.shape, (len(models), 19))
        self.assertEqual(np.isnan(coefficients).sum(), 2)
        self.assertTrue(np.isnan(detected).any())
        distances = pd.read_csv(ROOT / "results/emoji/inference/effect_profiles/emoji_mean_item_contrasts_clustering_distances.csv", index_col=0)
        common = coefficients[:, np.isfinite(coefficients).all(axis=0)]
        self.assertEqual(common.shape[1], 17)
        expected = np.sqrt(np.mean((common[:, None] - common[None, :]) ** 2, axis=2))
        np.testing.assert_allclose(distances.loc[models, models], expected, atol=1e-9)


if __name__ == "__main__":
    unittest.main()
