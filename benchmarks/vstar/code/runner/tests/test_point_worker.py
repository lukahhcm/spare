import unittest
from io import BytesIO
from unittest import mock

import numpy as np
from PIL import Image

from tool_server.tool_workers.online_workers.molmo_point_worker import MolmoPointWorker


class PointWorkerTest(unittest.TestCase):
    def test_annotated_image_keeps_original_size_and_can_be_reencoded(self):
        worker = MolmoPointWorker.__new__(MolmoPointWorker)
        image = Image.new("RGB", (64, 48), "white")

        annotated = worker.create_image_with_points(
            image,
            np.array([[32.0, 24.0]]),
            np.array([1], dtype=np.int32),
        )

        self.assertEqual(annotated.size, image.size)
        encoded = BytesIO()
        annotated.save(encoded, format="JPEG")
        decoded = Image.open(BytesIO(encoded.getvalue()))
        decoded.load()
        self.assertEqual(decoded.size, image.size)

    def test_legacy_molmo_disables_transformers_dynamic_cache(self):
        class LegacyMolmo:
            pass

        model = LegacyMolmo()
        MolmoPointWorker._disable_dynamic_cache(model)

        self.assertFalse(model._supports_default_dynamic_cache())

    def test_generation_config_enables_kv_cache_required_by_molmo(self):
        worker = MolmoPointWorker.__new__(MolmoPointWorker)
        worker.max_length = 200

        config = worker._generation_config()

        self.assertTrue(config.use_cache)
        self.assertEqual(config.max_new_tokens, 200)

    def test_generate_gate_does_not_resubmit_to_base_thread_pool(self):
        worker = MolmoPointWorker.__new__(MolmoPointWorker)
        worker.model_name = "Point"
        worker._generate_with_torch_inference = mock.Mock(
            return_value={"status": "success", "points": [{"x": 1, "y": 2}]}
        )
        worker.thread_pool = mock.Mock()
        worker.thread_pool.submit.side_effect = AssertionError(
            "generate_gate must not submit to the thread pool a second time"
        )

        result = worker.generate_gate({"image": "encoded", "description": "target"})

        self.assertEqual(result["status"], "success")
        worker._generate_with_torch_inference.assert_called_once_with(
            {"image": "encoded", "description": "target"}
        )
        worker.thread_pool.submit.assert_not_called()


if __name__ == "__main__":
    unittest.main()
