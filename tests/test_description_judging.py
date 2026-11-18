import contextlib
import io
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))
import judge_descriptions as judge
from plot_description_judgments import summarize


class DescriptionJudgingTest(unittest.TestCase):
    def test_summary_retains_versions_without_changing_scores(self):
        records = []
        for version, score in (("3", 0), ("4", 2)):
            result = {field: score for field in judge.COMPONENT_FIELDS}
            result.update(overall_score=score * 50, hallucination_penalty=0)
            records.append({"model": "m", "input_mode": "split", "description_prompt_version": version,
                            "judge_prompt_version": "4", "judgment_json": result})
        summary = summarize(records)[0]
        self.assertEqual((summary["n"], summary["mean_score"], summary["mean_visual_score"]), (2, 50, 50))
        self.assertEqual(summary["description_prompt_versions"], "3,4")
        self.assertEqual(summary["judge_prompt_versions"], "4")

    def test_resume_reuses_only_matching_successful_judgments(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1] / "tmp") as directory:
            root = Path(directory)
            row = {"model": "test-model", "input_mode": "split", "prompt_version": "4",
                   "source_video": "materials/test.m4v", "response": "first description"}
            gold = {"source_video": row["source_video"], "timeline": []}
            # The minimal gold fixture uses the legacy fallback fields.
            gold.update(text_segments=[], animation_descriptions=[])
            (root / "gold.json").write_text(json.dumps([gold]), encoding="utf-8")
            descriptions = root / "descriptions.jsonl"
            descriptions.write_text(json.dumps(row) + "\n", encoding="utf-8")
            args = SimpleNamespace(dry_run_prompts=False, gold=root / "gold.json", descriptions=[descriptions],
                                   input_modes=None, description_prompt_version="4", limit=None, resume=True,
                                   output=root / "judgments.jsonl", summary=root / "summary.csv", judge_model="gpt-4.1-mini")
            result = {field: 2 for field in judge.COMPONENT_FIELDS}
            result.update(hallucination_penalty=0, major_errors=[])
            with patch.object(judge, "parse_args", return_value=args), \
                    patch.object(judge, "judge_with_openai", return_value=json.dumps(result)) as api, \
                    patch.dict(os.environ, {"OPENAI_API_KEY": "test-fixture-not-a-real-key"}), \
                    contextlib.redirect_stdout(io.StringIO()):
                judge.main()
                judge.main()
                self.assertEqual(api.call_count, 1)
                self.assertEqual(len(judge.load_jsonl(args.output)), 1)
                row["response"] = "changed description"
                descriptions.write_text(json.dumps(row) + "\n", encoding="utf-8")
                judge.main()
                self.assertEqual(api.call_count, 2)
                self.assertEqual(len(judge.load_jsonl(args.output)), 2)


if __name__ == "__main__":
    unittest.main()
