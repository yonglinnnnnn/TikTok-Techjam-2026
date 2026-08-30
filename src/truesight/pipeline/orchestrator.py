from time import perf_counter

from ..utils import validate_image
from .adapters import PipelineComponents, Tier2Input
from .router import decide_route
from .result_schema import (
    FusionResult,
    TrueSightResult,
)


def run_pipeline(
    image_path: str,
    components: PipelineComponents | None = None,
) -> TrueSightResult:
    pipeline_started = perf_counter()
    validate_image(image_path)
    result = TrueSightResult.initialize(image_path=image_path)
    active_components = components or PipelineComponents.fake()

    tier1_analysis = active_components.tier1.analyze(image_path)
    result.tier1 = tier1_analysis.summary
    result.provenance = tier1_analysis.provenance
    result.forensics = tier1_analysis.forensics
    result.evidence.extend(tier1_analysis.evidence)

    route = decide_route(result.tier1)
    result.tier1.requires_tier2 = route.run_tier2

    if route.fast_path:
        result.is_ai_generated = True
        result.confidence = result.tier1.severity_weight
        result.source = tier1_analysis.source
        if not tier1_analysis.evidence:
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
        normalized_image_path = active_components.normalizer.normalize(
            image_path
        )

        if route.run_tier2:
            artifacts = result.forensics.get("artifacts", {})
            result.tier2 = active_components.tier2.analyze(
                Tier2Input(
                    image_path=normalized_image_path,
                    forensic_overlay=artifacts.get("vlm_overlay"),
                    candidate_regions=result.forensics.get(
                        "candidate_regions", []
                    ),
                    provenance_status=result.provenance.get("status"),
                    provenance_verified=result.tier1.provenance_verified,
                )
            )

        if route.run_tier3:
            result.tier3 = active_components.tier3.predict(
                normalized_image_path
            )

        vlm_confidence = (
            result.tier2.confidence if result.tier2 is not None else None
        )
        convnext_probability = (
            result.tier3.probability if result.tier3 is not None else None
        )

        # Temporary fusion. This deliberately does not add raw scores.
        result.confidence = convnext_probability
        result.is_ai_generated = (
            None
            if result.confidence is None
            else result.confidence >= 0.5
        )

        if result.tier2 is not None:
            result.source = result.tier2.source
            result.ai_coverage = result.tier2.ai_coverage
            result.evidence.extend(result.tier2.evidence)

        if result.tier3 is not None:
            result.heatmap_path = result.tier3.heatmap_path
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
                "vlm_confidence": vlm_confidence,
                "convnext_probability": convnext_probability,
            },
            latency_ms=0,
        )

    result.latency_ms = int(
        (perf_counter() - pipeline_started) * 1000
    )

    return result
