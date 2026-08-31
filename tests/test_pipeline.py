import json
import unittest
from pathlib import Path

from src.truesight.pipeline import Tier2Result, TrueSightResult, run_pipeline

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SAMPLE_IMAGE = PROJECT_ROOT / "data" / "samples" / "image_1.jpg"


class PipelineTests(unittest.TestCase):
    def test_blank_result_preserves_unknown_fields(self) -> None:
        result = TrueSightResult.initialize("example.jpg")

        self.assertIsNone(result.is_ai_generated)
        self.assertIsNone(result.confidence)
        self.assertIsNone(result.tier2)
        self.assertIsNone(result.tier3)
        self.assertIsNone(result.fusion)
        self.assertTrue(result.tier1.requires_tier2)

    def test_run_pipeline_returns_expected_result(self) -> None:
        result = run_pipeline(str(SAMPLE_IMAGE))

        self.assertIsInstance(result, TrueSightResult)
        self.assertEqual(result.image_path, str(SAMPLE_IMAGE))
        self.assertEqual(result.confidence, 0.82)
        self.assertTrue(result.is_ai_generated)
        self.assertIsNotNone(result.tier2)
        self.assertIsNotNone(result.tier3)
        self.assertIsNotNone(result.fusion)

    def test_result_can_be_serialized_to_dict(self) -> None:
        result = run_pipeline(str(SAMPLE_IMAGE))
        serialized = result.to_dict()

        json.dumps(serialized)

        self.assertEqual(serialized["image_path"], str(SAMPLE_IMAGE))
        self.assertEqual(serialized["schema_version"], "1.0")
        self.assertEqual(serialized["tier3"]["probability"], 0.82)
        self.assertTrue(serialized["is_ai_generated"])

    def test_invalid_probability_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            TrueSightResult(image_path="example.jpg", confidence=1.5)

        with self.assertRaises(ValueError):
            Tier2Result(confidence=-0.1)

    def test_missing_image_is_rejected(self) -> None:
        missing_image = PROJECT_ROOT / "data" / "samples" / "missing.jpg"

        with self.assertRaises(FileNotFoundError):
            run_pipeline(str(missing_image))


if __name__ == "__main__":
    unittest.main()
