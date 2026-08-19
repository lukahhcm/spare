import importlib.util
import os
from pathlib import Path
import sys
import unittest
from unittest import mock

from PIL import Image


RUNNER_PATH = Path(__file__).parents[1] / "scripts" / "run_tool_benchmarks.py"
sys.path.insert(0, str(RUNNER_PATH.parent))
SPEC = importlib.util.spec_from_file_location("run_tool_benchmarks_v5", RUNNER_PATH)
RUNNER = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(RUNNER)


class V5PruningPolicyTest(unittest.TestCase):
    def test_kl_visual_v5_hides_summary_deletes_low_kl_and_structures_high_kl(self):
        class FakeRuntime:
            tools = ["Point", "Crop", RUNNER.SUMMARY_TOOL_NAME]
            available_tools = tools

            @staticmethod
            def instructions():
                return "Point, Crop, summarize_the_task"

            @staticmethod
            def call_tool(tool_name, params, images):
                return {
                    "status": "success",
                    "message": f"verified {tool_name} evidence",
                    "error_code": 0,
                    "tool_response_from": tool_name,
                    "backend": "test",
                }, None

        low_kl_reasoning = "First grounded observation that KL should delete."
        high_kl_reasoning = (
            "I should keep planning prose out. The red sign is left of the blue door."
        )
        agent_outputs = iter(
            [
                f'<think>{low_kl_reasoning}</think>'
                '<tool_call>{"name":"Point","parameters":{"image":"img_1","description":"target"}}</tool_call>',
                f'<think>{high_kl_reasoning}</think>'
                '<tool_call>{"name":"Point","parameters":{"image":"img_1","description":"target detail"}}</tool_call>',
                '<think>Another local check is useful.</think>'
                '<tool_call>{"name":"Crop","parameters":{"image":"img_1","coordinates":[0,0,16,16]}}</tool_call>',
                '<think>The retained evidence supports option B.</think><response>B</response>',
            ]
        )
        third_agent_messages = []
        agent_calls = 0

        def fake_call_chat(base_url, model, messages, max_tokens, temperature, timeout):
            nonlocal agent_calls
            if messages[0]["content"] == "You are a faithful evidence compressor, not a problem-solving agent.":
                return '<response>{"facts":["verified target"],"next_decision":"inspect crop"}</response>'
            agent_calls += 1
            if agent_calls == 3:
                third_agent_messages.extend(messages)
            return next(agent_outputs)

        def fake_kl(**kwargs):
            segments = kwargs["segments"]
            return {
                "status": "ok",
                "prune_indices": [int(segments[0]["step_index"])],
                "segments": [],
                "raw_token_count": 2,
                "raw_rkl_min": 0.0,
                "raw_rkl_max": 1.0,
                "truncated": False,
            }

        env = {
            "BENCHMARK_V5_MIN_TOOL_CALLS": "2",
            "BENCHMARK_V5_GATE_ON_SUCCESSFUL_TOOLS": "1",
            "BENCHMARK_V5_MIN_REASONING_TOKENS": "1",
            "BENCHMARK_V5_MIN_SUCCESSFUL_TOOLS": "1",
            "BENCHMARK_V5_ALL_REASONING_CANDIDATES": "1",
            "BENCHMARK_V5_POST_FINAL_KL": "0",
            "BENCHMARK_V5_POST_ROLLOUT_KL": "0",
        }
        with mock.patch.dict(os.environ, env, clear=False), mock.patch.object(
            RUNNER, "call_chat", side_effect=fake_call_chat
        ), mock.patch.object(RUNNER, "log_summarization_kl", side_effect=fake_kl):
            result = RUNNER.run_item_with_tools(
                item={
                    "text": "Which option is correct? (A) no (B) yes",
                    "images": [Image.new("RGB", (32, 32), "white")],
                },
                runtime=FakeRuntime(),
                base_url="http://test",
                model="qwen3-vl-8b",
                max_rounds=6,
                min_tool_calls=1,
                max_tokens=256,
                temperature=0.0,
                timeout=5,
                enable_summary_kl_pruning=True,
                summary_mode="visual_state",
                v5_hidden_summary=True,
                v5_no_summary_context=True,
                task_name="vstar",
            )

        self.assertEqual(result["prediction"], "B")
        self.assertTrue(result["v5_no_summary_context"])
        self.assertTrue(result["v5_online_intervention"])
        self.assertTrue(third_agent_messages)
        next_context = "\n".join(str(message["content"]) for message in third_agent_messages)
        self.assertNotIn("Privileged auxiliary summary", next_context)
        self.assertNotIn("verified target", next_context)
        self.assertNotIn(low_kl_reasoning, next_context)
        self.assertNotIn(high_kl_reasoning, next_context)
        self.assertNotIn("I should keep planning prose out.", next_context)
        self.assertIn("Structured visual evidence kept:", next_context)
        self.assertIn("The red sign is left of the blue door.", next_context)

    def test_kl_delete_only_discards_summary_and_preserves_high_kl_reasoning(self):
        class FakeRuntime:
            tools = ["Point", "Crop", RUNNER.SUMMARY_TOOL_NAME]
            available_tools = tools

            @staticmethod
            def instructions():
                return "Point, Crop, summarize_the_task"

            @staticmethod
            def call_tool(tool_name, params, images):
                return {
                    "status": "success",
                    "message": f"verified {tool_name} evidence",
                    "error_code": 0,
                    "tool_response_from": tool_name,
                    "backend": "test",
                }, None

        first_reasoning = "First grounded observation that KL should delete."
        second_reasoning = "Second grounded observation that KL should retain verbatim."
        agent_outputs = iter(
            [
                f'<think>{first_reasoning}</think>'
                '<tool_call>{"name":"Point","parameters":{"image":"img_1","description":"target"}}</tool_call>',
                f'<think>{second_reasoning}</think>'
                '<tool_call>{"name":"Point","parameters":{"image":"img_1","description":"target detail"}}</tool_call>',
                '<think>Another local check is useful.</think>'
                '<tool_call>{"name":"Crop","parameters":{"image":"img_1","coordinates":[0,0,16,16]}}</tool_call>',
                '<think>The retained evidence supports option B.</think><response>B</response>',
            ]
        )
        third_agent_messages = []
        agent_calls = 0

        def fake_call_chat(base_url, model, messages, max_tokens, temperature, timeout):
            nonlocal agent_calls
            if messages[0]["content"] == "You are a faithful evidence compressor, not a problem-solving agent.":
                return '<response>{"facts":["verified target"],"next_decision":"inspect crop"}</response>'
            agent_calls += 1
            if agent_calls == 3:
                third_agent_messages.extend(messages)
            return next(agent_outputs)

        def fake_kl(**kwargs):
            segments = kwargs["segments"]
            return {
                "status": "ok",
                "prune_indices": [int(segments[0]["step_index"])],
                "segments": [],
                "raw_token_count": 2,
                "raw_rkl_min": 0.0,
                "raw_rkl_max": 1.0,
                "truncated": False,
            }

        env = {
            "BENCHMARK_V5_MIN_TOOL_CALLS": "2",
            "BENCHMARK_V5_GATE_ON_SUCCESSFUL_TOOLS": "1",
            "BENCHMARK_V5_MIN_REASONING_TOKENS": "1",
            "BENCHMARK_V5_MIN_SUCCESSFUL_TOOLS": "1",
            "BENCHMARK_V5_ALL_REASONING_CANDIDATES": "1",
            "BENCHMARK_V5_POST_FINAL_KL": "0",
            "BENCHMARK_V5_POST_ROLLOUT_KL": "0",
        }
        with mock.patch.dict(os.environ, env, clear=False), mock.patch.object(
            RUNNER, "call_chat", side_effect=fake_call_chat
        ), mock.patch.object(RUNNER, "log_summarization_kl", side_effect=fake_kl):
            result = RUNNER.run_item_with_tools(
                item={
                    "text": "Which option is correct? (A) no (B) yes",
                    "images": [Image.new("RGB", (32, 32), "white")],
                },
                runtime=FakeRuntime(),
                base_url="http://test",
                model="qwen3-vl-8b",
                max_rounds=6,
                min_tool_calls=1,
                max_tokens=256,
                temperature=0.0,
                timeout=5,
                enable_summary_kl_pruning=True,
                summary_mode="visual_state",
                v5_hidden_summary=True,
                v5_kl_delete_only=True,
                task_name="vstar",
            )

        self.assertEqual(result["prediction"], "B")
        self.assertTrue(result["v5_kl_delete_only"])
        self.assertTrue(result["v5_online_intervention"])
        self.assertTrue(third_agent_messages)
        next_context = "\n".join(str(message["content"]) for message in third_agent_messages)
        self.assertNotIn("Privileged auxiliary summary", next_context)
        self.assertNotIn("verified target", next_context)
        self.assertNotIn(first_reasoning, next_context)
        self.assertIn(second_reasoning, next_context)

    def test_hidden_v5_prunes_at_tool_gate_before_next_agent_decision(self):
        class FakeRuntime:
            tools = ["Point", "Crop", RUNNER.SUMMARY_TOOL_NAME]
            available_tools = tools

            @staticmethod
            def instructions():
                return "Point, Crop, summarize_the_task"

            @staticmethod
            def call_tool(tool_name, params, images):
                return {
                    "status": "success",
                    "message": f"verified {tool_name} evidence",
                    "error_code": 0,
                    "tool_response_from": tool_name,
                    "backend": "test",
                }, None

        agent_outputs = iter(
            [
                '<think>First grounded observation with enough visible reasoning.</think>'
                '<tool_call>{"name":"Point","parameters":{"image":"img_1","description":"target"}}</tool_call>',
                '<think>Second grounded observation with enough visible reasoning.</think>'
                '<tool_call>{"name":"Point","parameters":{"image":"img_1","description":"target detail"}}</tool_call>',
                '<think>The summary shows another local check is useful.</think>'
                '<tool_call>{"name":"Crop","parameters":{"image":"img_1","coordinates":[0,0,16,16]}}</tool_call>',
                '<think>The retained evidence supports option B.</think><response>B</response>',
            ]
        )
        third_agent_messages = []
        agent_calls = 0

        def fake_call_chat(base_url, model, messages, max_tokens, temperature, timeout):
            nonlocal agent_calls
            if messages[0]["content"] == "You are a faithful evidence compressor, not a problem-solving agent.":
                return '<response>{"facts":["verified target"],"next_decision":"inspect crop"}</response>'
            agent_calls += 1
            if agent_calls == 3:
                third_agent_messages.extend(messages)
            return next(agent_outputs)

        def fake_kl(**kwargs):
            segments = kwargs["segments"]
            return {
                "status": "ok",
                "prune_indices": [int(segments[0]["step_index"])],
                "segments": [],
                "raw_token_count": 2,
                "raw_rkl_min": 0.0,
                "raw_rkl_max": 1.0,
                "truncated": False,
            }

        env = {
            "BENCHMARK_V5_MIN_TOOL_CALLS": "2",
            "BENCHMARK_V5_GATE_ON_SUCCESSFUL_TOOLS": "1",
            "BENCHMARK_V5_MIN_REASONING_TOKENS": "1",
            "BENCHMARK_V5_MIN_SUCCESSFUL_TOOLS": "1",
            "BENCHMARK_V5_ALL_REASONING_CANDIDATES": "1",
            "BENCHMARK_V5_POST_FINAL_KL": "0",
            "BENCHMARK_V5_POST_ROLLOUT_KL": "0",
        }
        with mock.patch.dict(os.environ, env, clear=False), mock.patch.object(
            RUNNER, "call_chat", side_effect=fake_call_chat
        ), mock.patch.object(RUNNER, "log_summarization_kl", side_effect=fake_kl):
            result = RUNNER.run_item_with_tools(
                item={
                    "text": "Which option is correct? (A) no (B) yes",
                    "images": [Image.new("RGB", (32, 32), "white")],
                },
                runtime=FakeRuntime(),
                base_url="http://test",
                model="qwen3-vl-8b",
                max_rounds=6,
                min_tool_calls=1,
                max_tokens=256,
                temperature=0.0,
                timeout=5,
                enable_summary_kl_pruning=True,
                summary_mode="visual_state",
                v5_hidden_summary=True,
                task_name="vstar",
            )

        self.assertEqual(result["prediction"], "B")
        self.assertEqual(result["summary_tool_calls"], 1)
        self.assertEqual(result["v5_summary_trigger_round"], 2)
        self.assertTrue(result["v5_online_intervention"])
        self.assertEqual(result["post_summary_benchmark_tool_calls"], 1)
        self.assertTrue(third_agent_messages)
        self.assertIn("Privileged auxiliary summary", str(third_agent_messages[-1]["content"]))
        self.assertNotIn("First grounded observation", str(third_agent_messages[2]["content"]))

    def test_visual_backed_v5_keeps_prepruned_step_as_kl_candidate(self):
        self.assertTrue(
            RUNNER.v5_step_is_kl_candidate(
                step={"pruned": True, "reasoning": "The red sign is left of the door."},
                v5_hidden_summary=True,
                include_prepruned_override=False,
            )
        )

    def test_step_pruning_accounting_is_monotonic_and_capped(self):
        total, delta = RUNNER.updated_step_pruned_chars(
            original_reasoning_chars=100,
            replacement_reasoning_chars=20,
            previous_pruned_chars=0,
        )
        self.assertEqual((total, delta), (80, 80))

        total, delta = RUNNER.updated_step_pruned_chars(
            original_reasoning_chars=100,
            replacement_reasoning_chars=0,
            previous_pruned_chars=80,
        )
        self.assertEqual((total, delta), (100, 20))

        total, delta = RUNNER.updated_step_pruned_chars(
            original_reasoning_chars=100,
            replacement_reasoning_chars=0,
            previous_pruned_chars=100,
        )
        self.assertEqual((total, delta), (100, 0))

    def test_full_reasoning_prune_is_safe_when_actions_are_preserved(self):
        reason = RUNNER.v5_pruning_rejection_reason(
            kl_status="ok",
            successful_tool_calls=1,
            min_successful_tools=1,
            action_blocks_preserved=True,
        )

        self.assertIsNone(reason)

    def test_missing_verified_tool_evidence_is_rejected(self):
        reason = RUNNER.v5_pruning_rejection_reason(
            kl_status="ok",
            successful_tool_calls=0,
            min_successful_tools=1,
            action_blocks_preserved=True,
        )

        self.assertEqual(reason, "insufficient_successful_tool_evidence")

    def test_pruning_metrics_report_micro_and_macro_rates(self):
        metrics = RUNNER.pruning_fraction_metrics(
            pruned_token_counts=[90, 10, 0],
            before_token_counts=[100, 100, 0],
        )

        self.assertEqual(metrics["micro"], 0.5)
        self.assertAlmostEqual(metrics["macro_all"], 1.0 / 3.0)
        self.assertEqual(metrics["macro_nonempty"], 0.5)
        self.assertEqual(metrics["nonempty_rows"], 2)


if __name__ == "__main__":
    unittest.main()
