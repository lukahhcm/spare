import importlib.util
from pathlib import Path
import sys
import unittest


RUNNER_PATH = Path(__file__).parents[1] / "scripts" / "run_tool_benchmarks.py"
sys.path.insert(0, str(RUNNER_PATH.parent))
SPEC = importlib.util.spec_from_file_location("run_tool_benchmarks", RUNNER_PATH)
RUNNER = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(RUNNER)


class ToolParameterRepairTest(unittest.TestCase):
    def test_out_of_range_image_id_is_repaired_to_latest_available_image(self):
        params, repaired = RUNNER.repair_tool_parameters(
            "Point",
            {"image": "img_9", "description": "the shovel"},
            "Where is the shovel?",
            available_image_count=1,
        )

        self.assertTrue(repaired)
        self.assertEqual(params["image"], "img_1")

    def test_point_description_is_filled_from_question_targets(self):
        params, repaired = RUNNER.repair_tool_parameters(
            "Point",
            {"image": "img_1"},
            "What is the material of the glove?",
        )

        self.assertTrue(repaired)
        self.assertEqual(
            params,
            {
                "image": "img_1",
                "description": "locate the glove material",
            },
        )

    def test_existing_point_description_is_preserved(self):
        params, repaired = RUNNER.repair_tool_parameters(
            "Point",
            {"image": "img_1", "description": "the red glove"},
            "What is the material of the glove?",
        )

        self.assertFalse(repaired)
        self.assertEqual(params["description"], "the red glove")

    def test_non_point_parameters_are_unchanged(self):
        original = {"image": "img_1"}
        params, repaired = RUNNER.repair_tool_parameters(
            "OCR",
            original,
            "What text is visible?",
        )

        self.assertFalse(repaired)
        self.assertEqual(params, original)
        self.assertIsNot(params, original)


if __name__ == "__main__":
    unittest.main()
