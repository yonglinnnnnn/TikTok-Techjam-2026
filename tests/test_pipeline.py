import unittest
from pathlib import Path

from src.truesight.pipeline import TrueSightResult, run_pipeline

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SAMPLE_IMAGE = PROJECT_ROOT / "data" / "samples" / "image_1.jpg"


class PipelineTests(unittest.TestCase):
    def test_run_pipeline_returns_expected_result(self) -> None:
        result = run_pipeline(str(SAMPLE_IMAGE))

        self.assertIsInstance(result, TrueSightResult)
        self.assertEqual(result.image_path, str(SAMPLE_IMAGE))
        self.assertEqual(result.final_confidence, 0.82)
        self.assertTrue(result.final_is_ai_generated)

    def test_result_can_be_serialized_to_dict(self) -> None:
        result = run_pipeline(str(SAMPLE_IMAGE))
        serialized = result.to_dict()

        self.assertEqual(serialized["image_path"], str(SAMPLE_IMAGE))
        self.assertEqual(serialized["model"]["confidence"], 0.82)
        self.assertTrue(serialized["model"]["is_ai_generated"])

    def test_missing_image_is_rejected(self) -> None:
        missing_image = PROJECT_ROOT / "data" / "samples" / "missing.jpg"

        with self.assertRaises(FileNotFoundError):
            run_pipeline(str(missing_image))


if __name__ == "__main__":
    unittest.main()
