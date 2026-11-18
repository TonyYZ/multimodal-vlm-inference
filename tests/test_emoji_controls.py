import csv
import json
import sys
import unittest
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
from prepare_emoji_pilot import posttext_controls
from plot_emoji_pilot_barcharts import draw_panel, values
from compare_model_performance import parse_score


class EmojiPremiseControlsTest(unittest.TestCase):
    def setUp(self):
        self.sequences = [json.loads(line) for line in (ROOT / "processed/emoji_pilot/index.jsonl").read_text(encoding="utf-8").splitlines()]
        self.statements = json.loads((ROOT / "code/emoji_pilot_statements.json").read_text(encoding="utf-8"))
        with (ROOT / "materials/emoji_study/emoji_pilot_manifest.csv").open(encoding="utf-8", newline="") as stream:
            self.rows = list(csv.DictReader(stream))

    def test_controls_keep_questions_and_do_not_invent_human_data(self):
        sequences, questions, rows = posttext_controls(self.sequences, self.statements, self.rows)
        self.assertEqual((len(sequences), len(rows)), (5, 10))
        for row in rows:
            self.assertEqual(questions[row["source_video"]], self.statements[row["target_source"]])
            self.assertEqual(row["human_mean"], "")
            self.assertFalse(row["human_available"])
            self.assertEqual(row["human_n"], 0)

    def test_cigarette_matches_lyns_example(self):
        _, _, rows = posttext_controls(self.sequences, self.statements, self.rows)
        row = next(row for row in rows if row["emoji"] == "Cigarette")
        self.assertEqual(row["sentence"], "If the employee steps out onto the balcony to do this: \U0001f6ac, he will miss a phone call.")

    def test_missing_human_controls_do_not_shift_bar_positions(self):
        cells = [("Cigarette", "Target premise"), ("Cigarette", "Control premise")]
        rows = [{"experiment": "Experiment 1", "environment": emoji,
                 "inference_type": premise, "material_condition": kind,
                 "model_score": 50, "human_mean": 90 if premise == "Target premise" else np.nan}
                for emoji, premise in cells for kind in ("Target", "Control")]
        fig, axes = plt.subplots(2, 1)
        try:
            for ax, field in zip(axes, ("human_mean", "model_score")):
                draw_panel(ax, rows, cells, field, "", "Inference", False)
            self.assertEqual(axes[0].get_xlim(), (-0.6, 1.6))
            self.assertEqual(axes[0].get_xlim(), axes[1].get_xlim())
            np.testing.assert_allclose([p.get_x() for p in axes[0].patches],
                                       [p.get_x() for p in axes[1].patches])
        finally:
            plt.close(fig)

    def test_full_materials_cover_original_human_items(self):
        self.assertEqual(len(self.sequences), 140)
        self.assertEqual(len(self.rows), 190)
        for experiment, count, items in (("Experiment 1", 90, 5), ("Experiment 2", 90, 9), ("Experiment 3", 10, 5)):
            rows = [r for r in self.rows if r["experiment"] == experiment]
            self.assertEqual(len(rows), count)
            self.assertEqual(len({r["emoji"] for r in rows}), items)
            self.assertTrue(all(int(r["human_n"]) > 0 for r in rows))
            self.assertTrue(all(np.isfinite(float(r["human_mean"])) for r in rows))

    def test_bar_values_average_items_instead_of_retaining_last(self):
        rows = [{"experiment": "Experiment 2", "environment": "Might", "statement_type": "target", "model_score": v}
                for v in (0, 40, 80)]
        self.assertEqual(values(rows, [("Might", "")], "model_score", "Target"), [40])
        rows += [{**rows[0], "environment": "Question"} for _ in range(2)]
        self.assertTrue(np.isnan(values(rows, [("Question", "")], "model_score", "Target")[0]))

    def test_full_run_saved_responses_and_prompts_match_materials(self):
        ratings = {}
        for path in sorted((ROOT / "results/emoji/inference/ratings").glob("*.jsonl")):
            for line in path.read_text(encoding="utf-8").splitlines():
                row = json.loads(line)
                ratings[row["model"], row["source_video"], row["statement_type"]] = row
        models = {model for model, _, _ in ratings}
        self.assertGreaterEqual(len(models), 7)
        sequences = {item["source_video"]: item for item in self.sequences}
        invalid = []
        for model in models:
            for material in self.rows:
                source, kind = material["source_video"], material["statement_type"]
                row = ratings[model, source, kind]
                self.assertFalse(row.get("error"))
                if parse_score(row.get("response")) is None:
                    invalid.append((model, source))
                self.assertEqual(row["input_mode"], "pure-text")
                self.assertEqual(row["content"][0], sequences[source]["qwen_content"][0])
                self.assertEqual(" ".join(row["statement"].split()), " ".join(material["inference"].split()))
        self.assertEqual(invalid, [("Qwen/Qwen3-VL-2B-Instruct",
                                    "materials\\emoji_study\\Exp1_CoText_Phone_Control_Exactly-one_Existential.txt")])


if __name__ == "__main__":
    unittest.main()
