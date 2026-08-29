from time import perf_counter

from ..utils import validate_image
from .router import decide_route
from .result_schema import (
    FusionResult,
    Tier1Result,
    Tier2Result,
    Tier3Result,
    TrueSightResult,
)


def run_pipeline(image_path: str) -> TrueSightResult:
    pipeline_started = perf_counter()
    validate_image(image_path)
    result = TrueSightResult.initialize(image_path=image_path)

    # Placeholder for actual pipeline logic
    # In a real implementation, this function would run the image through
    # various models and return the results.
    
    # Example dummy results for demonstration purposes
    # Fake Tier 1 result
    result.provenance = {
        "status": "not_present",
        "c2pa": None,
        "openai": None,
        "metadata": {},
    }

    result.forensics = {
        "candidate_regions": [],
        "artifacts": {},
    }

    result.tier1 = Tier1Result(
        watermark_detected=None,
        provenance_verified=False,
        verified_ai_signal=False,
        verified_capture_signal=False,
        severity_weight=0.0,
        severity_calibration={
            "calibrated": False,
            "method": "policy_v1",
        },
        forensic_integrity_weight=0.0,
        requires_tier2=True,
    )

    route = decide_route(result.tier1)
    result.tier1.requires_tier2 = route.run_tier2

    if route.fast_path:
        result.is_ai_generated = True
        result.confidence = result.tier1.severity_weight
        result.evidence.append("Verified AI provenance signal detected")

        result.fusion = FusionResult(
            method="verified_provenance_fast_path",
            calibrated=False,
            decision_threshold=0.5,
            inputs={
                "provenance_severity": result.tier1.severity_weight,
                "provenance_verified_ai": True,
            },
        )
    else:
        # Fake Tier 2 result.
        result.tier2 = Tier2Result(
            is_ai_generated=True,
            confidence=0.70,
            source="Stable Diffusion",
            ai_coverage=0.65,
            evidence=[
                "Synthetic-looking texture patterns were detected"
            ],
            latency_ms=0,
        )

        # Fake Tier 3 result.
        result.tier3 = Tier3Result(
            probability=0.82,
            heatmap_path=None,
            evidence=["ConvNeXt visual classifier completed"],
            latency_ms=0,
        )

        # Temporary fusion. This deliberately does not add raw scores.
        result.confidence = result.tier3.probability
        result.is_ai_generated = (
            result.confidence is not None
            and result.confidence >= 0.5
        )
        result.source = result.tier2.source
        result.ai_coverage = result.tier2.ai_coverage
        result.heatmap_path = result.tier3.heatmap_path

        result.evidence.extend(result.tier2.evidence)
        result.evidence.extend(result.tier3.evidence)

        result.fusion = FusionResult(
            method="temporary_convnext_only",
            calibrated=False,
            decision_threshold=0.5,
            inputs={
                "provenance_severity": result.tier1.severity_weight,
                "provenance_verified_ai": (
                    result.tier1.verified_ai_signal
                ),
                "forensic_integrity": (
                    result.tier1.forensic_integrity_weight
                ),
                "vlm_confidence": result.tier2.confidence,
                "convnext_probability": result.tier3.probability,
            },
            latency_ms=0,
        )

    result.latency_ms = int(
        (perf_counter() - pipeline_started) * 1000
    )

    return result
