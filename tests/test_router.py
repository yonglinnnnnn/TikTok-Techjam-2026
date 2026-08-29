import unittest

from src.truesight.pipeline import Tier1Result, decide_route


class RouterTests(unittest.TestCase):
    def test_verified_ai_signal_uses_fast_path(self) -> None:
        tier1 = Tier1Result(
            provenance_verified=True,
            verified_ai_signal=True,
            severity_weight=0.98,
            requires_tier2=False,
        )

        route = decide_route(tier1)

        self.assertTrue(route.fast_path)
        self.assertFalse(route.run_tier2)
        self.assertFalse(route.run_tier3)
        self.assertEqual(route.reason, "verified_ai_signal")

    def test_missing_provenance_continues_downstream(self) -> None:
        route = decide_route(Tier1Result())

        self.assertFalse(route.fast_path)
        self.assertTrue(route.run_tier2)
        self.assertTrue(route.run_tier3)

    def test_unverified_detection_does_not_use_fast_path(self) -> None:
        tier1 = Tier1Result(
            watermark_detected=True,
            provenance_verified=False,
            verified_ai_signal=False,
            severity_weight=0.15,
        )

        route = decide_route(tier1)

        self.assertFalse(route.fast_path)
        self.assertTrue(route.run_tier2)
        self.assertTrue(route.run_tier3)

    def test_verified_capture_remains_inconclusive(self) -> None:
        tier1 = Tier1Result(
            provenance_verified=True,
            verified_capture_signal=True,
        )

        route = decide_route(tier1)

        self.assertFalse(route.fast_path)
        self.assertTrue(route.run_tier2)
        self.assertTrue(route.run_tier3)


if __name__ == "__main__":
    unittest.main()
