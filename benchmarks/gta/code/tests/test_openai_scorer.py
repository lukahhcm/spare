from __future__ import annotations

import os
import unittest

from reasoning_prune.openai_scorer import OpenAICompletionScorer


@unittest.skipUnless(os.environ.get("RUN_LOCAL_VLLM_TEST") == "1", "requires a local vLLM endpoint")
class OpenAIScorerTest(unittest.TestCase):
    def test_returns_continuation_prompt_positions(self):
        scorer = OpenAICompletionScorer("http://127.0.0.1:8001/v1", "qwen3-vl-8b", top_logprobs=5)
        positions = scorer.score("System:\n", "reasoning tokens here")
        self.assertGreater(len(positions), 0)
        self.assertTrue(all(position for position in positions))


if __name__ == "__main__":
    unittest.main()
