from __future__ import annotations
from typing import Any
from .base_provider import VLMResult

REQUIRED_KEYS = {"source_estimate", "ai_coverage", "confidence", "reasoning", "evidence"}
VALID_SOURCES = {"DALL-E", "Midjourney", "Stable Diffusion", "Other AI", "Real", "Uncertain"}


def _clamp01(value: Any, default: float = 0.5) -> float:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return default
    return max(0.0, min(1.0, v))


def parse_vlm_response(raw: dict[str, Any], provider: str) -> VLMResult:
    if not isinstance(raw, dict):
        return VLMResult(
            provider=provider, source_estimate="Uncertain", ai_coverage=None,
            confidence=0.0, reasoning="Provider returned a non-JSON-object response.",
            evidence=[], raw_response=None,
        )

    missing = REQUIRED_KEYS - raw.keys()
    source = raw.get("source_estimate")
    if source not in VALID_SOURCES:
        source = "Uncertain"

    evidence = raw.get("evidence")
    if not isinstance(evidence, list):
        evidence = []
    evidence = [str(e) for e in evidence][:8]

    reasoning = raw.get("reasoning")
    if not isinstance(reasoning, str) or not reasoning.strip():
        reasoning = "No reasoning provided." if not missing else (
            f"Partial response — missing fields: {sorted(missing)}."
        )

    return VLMResult(
        provider=provider,
        source_estimate=source,
        ai_coverage=_clamp01(raw.get("ai_coverage"), default=0.5) if "ai_coverage" in raw else None,
        confidence=_clamp01(raw.get("confidence"), default=0.0),
        reasoning=reasoning,
        evidence=evidence,
        raw_response=raw,
    )