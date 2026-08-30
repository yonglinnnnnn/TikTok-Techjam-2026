"""
Runs the Tier 2 VLM pipeline against the sample built by sample_sid_set.py
and reports how well source_estimate / ai_coverage line up with SID_Set's
ground-truth labels — this is calibration/debugging for YOUR tier, not the
final pipeline evaluation (that's evaluate.py / Member 4 & 5's job, which
combines all 3 tiers).

Usage:
    python scripts/calibrate_vlm.py --providers openai gemini
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from truesight.vlm import GeminiProvider, OpenAIProvider, run_vlm_tier

MANIFEST_PATH = Path("tests/fixtures/sample_images/manifest.json")
REPORT_PATH = Path("outputs/metrics/vlm_calibration_report.json")

PROVIDER_MAP = {"openai": OpenAIProvider, "gemini": GeminiProvider}

# SID_Set label -> whether we'd expect the VLM to flag this as AI-touched.
# tampered (2) is partially AI-edited, so "expected AI" too, just with
# lower expected ai_coverage than full_synthetic (1).
EXPECTED_AI = {0: False, 1: True, 2: True}


async def run(providers: list[str]) -> None:
    if not MANIFEST_PATH.exists():
        raise SystemExit(
            f"{MANIFEST_PATH} not found — run scripts/sample_sid_set.py first."
        )

    manifest = json.loads(MANIFEST_PATH.read_text())
    provider_instances = [PROVIDER_MAP[p]() for p in providers]

    results = []
    correct = 0

    for entry in manifest:
        tier_result = await run_vlm_tier(entry["image_path"], provider_instances)
        predicted_ai = tier_result.source not in (None, "Real", "Uncertain")
        is_correct = predicted_ai == EXPECTED_AI[entry["label"]]
        correct += int(is_correct)

        results.append({
            "image_path": entry["image_path"],
            "ground_truth": entry["label_name"],
            "predicted_source": tier_result.source,
            "predicted_ai_coverage": tier_result.ai_coverage,
            "confidence": tier_result.confidence,
            "correct": is_correct,
            "evidence": tier_result.evidence,
            "per_provider_errors": [
                {"provider": r.provider, "error": r.error}
                for r in tier_result.per_provider if r.error
            ],
        })
        status = "correct" if is_correct else "WRONG"
        print(f"[{status}] {entry['label_name']:>15} -> "
              f"{tier_result.source} (coverage={tier_result.ai_coverage})  "
              f"{entry['image_path']}")

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps({
        "accuracy": correct / len(manifest) if manifest else 0.0,
        "n_images": len(manifest),
        "providers": providers,
        "results": results,
    }, indent=2))

    print(f"\nAccuracy: {correct}/{len(manifest)} "
          f"({100 * correct / len(manifest):.1f}%)")
    print(f"Full report: {REPORT_PATH}")
    print("\nUse the 'WRONG' rows above as candidates for your Devpost "
          "case-study section (e.g. tampered images the VLM missed, or "
          "real images it flagged as AI).")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--providers", nargs="+", default=["openai"],
                         choices=list(PROVIDER_MAP), help="Providers to test")
    args = parser.parse_args()
    asyncio.run(run(args.providers))