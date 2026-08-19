import importlib.util
from pathlib import Path
import unittest


SCRIPT_PATH = Path(__file__).parents[1] / "scripts" / "merge_v5_visual_fallback.py"
SPEC = importlib.util.spec_from_file_location("merge_v5_visual_fallback", SCRIPT_PATH)
MERGER = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MERGER)


class MergeV5VisualFallbackTest(unittest.TestCase):
    def test_disagreement_falls_back_without_consulting_candidate_score(self):
        candidate = {
            "idx": "1",
            "prediction": "B",
            "score": 1.0,
            "summary_kl_pruned_reasoning_tokens_est": 90,
            "reasoning_tokens_before_prune_est": 100,
        }
        visual = {
            "idx": "1",
            "prediction": "A",
            "score": 0.0,
            "summary_kl_pruned_reasoning_tokens_est": 40,
            "reasoning_tokens_before_prune_est": 100,
        }

        merged = MERGER.merge_candidate_with_visual(candidate, visual)

        self.assertEqual(merged["prediction"], "A")
        self.assertEqual(merged["score"], 0.0)
        self.assertTrue(merged["v5_visual_fallback"])
        self.assertEqual(merged["v5_candidate_prediction"], "B")

    def test_equivalent_wrapped_answer_keeps_v5_candidate(self):
        candidate = {"idx": "2", "prediction": "\\boxed{A}", "score": 1.0}
        visual = {"idx": "2", "prediction": "<response>A</response>", "score": 1.0}

        merged = MERGER.merge_candidate_with_visual(candidate, visual)

        self.assertEqual(merged["prediction"], "\\boxed{A}")
        self.assertFalse(merged["v5_visual_fallback"])


if __name__ == "__main__":
    unittest.main()
