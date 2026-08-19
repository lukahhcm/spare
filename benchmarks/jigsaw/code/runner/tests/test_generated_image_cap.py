import importlib.util
from pathlib import Path
import sys
import unittest


RUNNER_PATH = Path(__file__).parents[1] / "scripts" / "run_tool_benchmarks.py"
sys.path.insert(0, str(RUNNER_PATH.parent))
SPEC = importlib.util.spec_from_file_location("run_tool_benchmarks_image_cap", RUNNER_PATH)
RUNNER = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(RUNNER)


class GeneratedImageCapTest(unittest.TestCase):
    def test_unlimited_mode_keeps_generated_images(self):
        self.assertTrue(RUNNER.should_append_generated_image(8, 1, 0))

    def test_cap_blocks_only_after_generated_image_limit(self):
        self.assertTrue(RUNNER.should_append_generated_image(3, 1, 3))
        self.assertFalse(RUNNER.should_append_generated_image(4, 1, 3))


if __name__ == "__main__":
    unittest.main()
