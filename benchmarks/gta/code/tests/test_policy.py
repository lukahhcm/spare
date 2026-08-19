from __future__ import annotations

import unittest

from reasoning_prune.policy import PruneConfig, ReasoningSegment, SummaryKLPruner


class FakeScorer:
    def score(self, prompt: str, continuation: str):
        if prompt == "student":
            return [{"a": -0.1}, {"b": -0.1}, {"c": -0.1}, {"d": -0.1}]
        return [{"a": -0.1}, {"b": -0.1}, {"c": -4.0}, {"d": -4.0}]


class PolicyTest(unittest.TestCase):
    def test_v2_keeps_action_exact_and_prunes_covered_segment(self):
        first = ReasoningSegment("one", "obsolete prose", '<tool_call>{"x": 1}</tool_call>', 0, {"reasoning_token_count": 2})
        second = ReasoningSegment("two", "important coordinates (1, 2)", '<tool_call>{"x": 2}</tool_call>', 1, {"reasoning_token_count": 2})
        event = SummaryKLPruner(FakeScorer(), PruneConfig(threshold=0.2, min_high_tokens=1)).prune(
            [first, second], "summary", "student", "teacher"
        )
        self.assertTrue(first.pruned)
        self.assertFalse(second.pruned)
        self.assertEqual(event.decisions[0]["replacement"], '<tool_call>{"x": 1}</tool_call>')
        self.assertTrue(event.decisions[1]["keep"])

    def test_v3_keeps_evidence_without_reformatting_action(self):
        segment = ReasoningSegment(
            "one", "OCR text Elm Street is at coordinate (245, 380).", "<native-action foo='bar'/>", 0,
            {"reasoning_token_count": 2},
        )
        class CoveredScorer:
            def score(self, prompt, continuation):
                return [{"a": -0.1}, {"b": -0.1}]

        event = SummaryKLPruner(CoveredScorer(), PruneConfig(evidence_mode="v3", threshold=0.2, min_high_tokens=1)).prune(
            [segment], "summary", "student", "teacher"
        )
        self.assertTrue(segment.pruned)
        self.assertTrue(str(event.decisions[0]["replacement"]).endswith("<native-action foo='bar'/>"))
        self.assertIn("Elm Street", str(event.decisions[0]["replacement"]))

    def test_missing_boundaries_fails_closed(self):
        segment = ReasoningSegment("one", "prose", "<call/>", 0)
        event = SummaryKLPruner(FakeScorer()).prune([segment], "summary", "student", "teacher")
        self.assertEqual(event.skipped_reason, "missing_token_boundaries")
        self.assertFalse(segment.pruned)


if __name__ == "__main__":
    unittest.main()
