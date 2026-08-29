from .result_schema import (
    ModelResult,
    ProvenanceResult,
    TrueSightResult,
    VLMResult,
)

from ..utils import validate_image


def run_pipeline(image_path: str) -> TrueSightResult:
    validate_image(image_path)

    # Placeholder for actual pipeline logic
    # In a real implementation, this function would run the image through
    # various models and return the results.
    
    # Example dummy results for demonstration purposes
    # Fake Tier 1 result
    provenance = ProvenanceResult(
        detected=False,
        source=None,
        method=None,
        confidence=None,
    )

    # Fake Tier 2 result
    vlm = VLMResult(
        source_estimate="Stable Diffusion",
        confidence=0.70,
        explanation="Some synthetic-looking texture patterns were detected.",
    )

    # Fake Tier 3 result
    model = ModelResult(
        confidence=0.82,
        is_ai_generated=True,
        heatmap_path=None,
    )

    # Temporary fusion logic
    if provenance.detected:
        final_confidence = (
            provenance.confidence
            if provenance.confidence is not None
            else 0.99
        )
        final_is_ai_generated = True
    else:
        final_confidence = model.confidence
        final_is_ai_generated = model.is_ai_generated

    return TrueSightResult(
        image_path=image_path,
        provenance=provenance,
        vlm=vlm,
        model=model,
        final_confidence=final_confidence,
        final_is_ai_generated=final_is_ai_generated,
    )
