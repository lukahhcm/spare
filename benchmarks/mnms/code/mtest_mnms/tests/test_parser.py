import unittest

from mnms.parser import Parser


class ParserTest(unittest.TestCase):
    def test_escaped_underscore_is_normalized(self):
        result = Parser("multi-step").parse(
            '{"nodes": [{"id": 0, "name": "image\\_captioning", "args": {}}]}'
        )

        self.assertTrue(result["status"])
        self.assertEqual(result["content"][0]["name"], "image_captioning")


if __name__ == "__main__":
    unittest.main()
