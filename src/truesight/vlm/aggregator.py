from __future__ import annotations

import asyncio
from dataclasses import dataclass

from .base_provider import BaseVLMProvider, VLMResult


@dataclass
class VLMTierResult:
    source: str | None
    ai_coverage: float | None
    confidence: float | None
    evidence: list[str]
    reasoning: str | None
    per_provider: list[VLMResult]


def _merge(results: list[VLMResult]) -> VLMTierResult:
    valid = [r for r in results if r.is_valid()]
    if not valid:
        return VLMTierResult(
            source="Uncertain", ai_coverage=None, confidence=0.0,
            evidence=["all VLM providers failed or returned invalid output"],
            reasoning=None, per_provider=results,
        )

    weighted_sum, weight_total = 0.0, 0.0
    for r in valid:
        if r.ai_coverage is not None:
            w = r.confidence or 0.01
            weighted_sum += r.ai_coverage * w
            weight_total += w
    ai_coverage = (weighted_sum / weight_total) if weight_total else None

    best = max(valid, key=lambda r: r.confidence or 0.0)
    evidence = [e for r in valid for e in r.evidence]

    return VLMTierResult(
        source=best.source_estimate, ai_coverage=ai_coverage, confidence=best.confidence,
        evidence=evidence, reasoning=best.reasoning, per_provider=results,
    )


async def run_vlm_tier(image_path: str, providers: list[BaseVLMProvider]) -> VLMTierResult:
    results = await asyncio.gather(*(p.analyze(image_path) for p in providers))
    return _merge(list(results))