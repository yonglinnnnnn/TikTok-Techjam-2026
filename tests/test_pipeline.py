import unittest

from src.pipeline import TrueSightResult, run_pipeline


class PipelineTests(unittest.TestCase):
    def test_run_pipeline_returns_expected_result(self) -> None:
        result = run_pipeline("test.jpg")

        self.assertIsInstance(result, TrueSightResult)
        self.assertEqual(result.image_path, "test.jpg")
        self.assertEqual(result.final_confidence, 0.82)
        self.assertTrue(result.final_is_ai_generated)

    def test_result_can_be_serialized_to_dict(self) -> None:
        result = run_pipeline("test.jpg")
        serialized = result.to_dict()

        self.assertEqual(serialized["image_path"], "test.jpg")
        self.assertEqual(serialized["model"]["confidence"], 0.82)
        self.assertTrue(serialized["model"]["is_ai_generated"])


if __name__ == "__main__":
    unittest.main()
