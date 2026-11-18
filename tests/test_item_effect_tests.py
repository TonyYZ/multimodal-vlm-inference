import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
from item_effect_tests import signed_rank


class ItemWilcoxonTest(unittest.TestCase):
    def test_constant_nonzero_contrasts_have_valid_exact_p(self):
        for n, expected in ((2, .5), (3, .25), (4, .125), (5, .0625)):
            result = signed_rank([100] * n)
            self.assertEqual(result["p_value"], expected)
            self.assertEqual(result["estimate"], 100)

    def test_all_zero_and_tied_differences(self):
        self.assertEqual(signed_rank([0, 0, 0])["p_value"], 1)
        self.assertEqual(signed_rank([1, -1, 0])["p_value"], 1)
        self.assertEqual(signed_rank([0, 1, 1])["p_value"], .5)

    def test_missing_is_not_zero(self):
        self.assertTrue(np.isnan(signed_rank([1, 2, np.nan])["p_value"]))

    def test_one_did_per_item_not_one_per_cell(self):
        scores = pd.read_csv(ROOT / "results/animation/human_tests/per_item_scores.csv", keep_default_na=False)
        scores = scores[scores.model.eq("google/gemma-4-12B-it") & scores.input_format.eq("split")]
        cells = scores[scores.condition.eq("Scalar Implicature") & scores.environment.eq("Positive")]
        self.assertEqual(len(cells), 12)
        self.assertEqual(cells.item.nunique(), 3)
        items = pd.read_csv(ROOT / "results/animation/human_tests/per_item_contrasts.csv", keep_default_na=False)
        observed = items[items.model.eq("google/gemma-4-12B-it") & items.input_format.eq("split") &
                         items.condition.eq("Scalar Implicature") & items.environment.eq("Positive")]
        for item, group in cells.groupby("item"):
            table = group.set_index(["premise", "inference"]).score
            expected = table["Target", "Target"] - table["Target", "Baseline"] - table["Control", "Target"] + table["Control", "Baseline"]
            self.assertEqual(observed.loc[observed.item.astype(str).eq(str(item)), "difference"].iloc[0], expected)

    def test_primary_p_values_use_the_item_contrasts(self):
        for study in ("animation", "gesture", "emoji"):
            base = ROOT / "results" / study
            path = base / "inference/effect_profiles"
            if study != "emoji":
                path /= "item_effects"
            tests = pd.read_csv(path / "item_wilcoxon_tests.csv", keep_default_na=False, na_values=["NA"])
            items = pd.read_csv(base / "human_tests/per_item_contrasts.csv", keep_default_na=False, na_values=["NA"])
            for row in tests.itertuples():
                values = items[items.model.eq(row.model) & items.input_format.eq(row.input_format) & items.test.eq(row.test)].difference
                expected = signed_rank(values)
                np.testing.assert_allclose([row.estimate, row.p_value], [expected["estimate"], expected["p_value"]], equal_nan=True)


if __name__ == "__main__":
    unittest.main()
