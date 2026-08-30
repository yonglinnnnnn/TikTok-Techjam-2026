from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from .result_schema import (
    Tier1Result,
    Tier2Result,
    Tier3Result,
    default_forensics,
    default_provenance,
)


@dataclass
class Tier1Analysis:
    """Normalized hand-off from provenance analysis to orchestration."""

    summary: Tier1Result
    provenance: dict[str, Any] = field(default_factory=default_provenance)
    forensics: dict[str, Any] = field(default_factory=default_forensics)
    evidence: list[str] = field(default_factory=list)
    source: str | None = None


@dataclass(frozen=True)
class Tier2Input:
    """The deliberately limited context supplied to the VLM tier."""

    image_path: str
    forensic_overlay: str | None
    candidate_regions: list[dict[str, Any]]
    provenance_status: str | None
    provenance_verified: bool


class Tier1Adapter(Protocol):
    def analyze(self, original_image_path: str) -> Tier1Analysis:
        """Inspect the untouched uploaded image and return Tier 1 output."""


class Tier2Adapter(Protocol):
    def analyze(self, inputs: Tier2Input) -> Tier2Result:
        """Analyze the normalized image and concise supporting context."""


class Tier3Adapter(Protocol):
    def predict(self, normalized_image_path: str) -> Tier3Result:
        """Run the visual classifier on only the normalized RGB image."""


class ImageNormalizer(Protocol):
    def normalize(self, original_image_path: str) -> str:
        """Return a normalized derivative without modifying the original."""


class PassthroughNormalizer:
    """Development fallback used until image normalization is connected."""

    def normalize(self, original_image_path: str) -> str:
        return original_image_path


class FakeTier1Adapter:
    def analyze(self, original_image_path: str) -> Tier1Analysis:
        del original_image_path
        return Tier1Analysis(summary=Tier1Result())


class FakeTier2Adapter:
    def analyze(self, inputs: Tier2Input) -> Tier2Result:
        del inputs
        return Tier2Result(
            is_ai_generated=True,
            confidence=0.70,
            source="Stable Diffusion",
            ai_coverage=0.65,
            evidence=[
                "Synthetic-looking texture patterns were detected"
            ],
            latency_ms=0,
        )


class FakeTier3Adapter:
    def predict(self, normalized_image_path: str) -> Tier3Result:
        del normalized_image_path
        return Tier3Result(
            probability=0.82,
            heatmap_path=None,
            evidence=["ConvNeXt visual classifier completed"],
            latency_ms=0,
        )


@dataclass
class PipelineComponents:
    tier1: Tier1Adapter
    tier2: Tier2Adapter
    tier3: Tier3Adapter
    normalizer: ImageNormalizer

    @classmethod
    def fake(cls) -> PipelineComponents:
        return cls(
            tier1=FakeTier1Adapter(),
            tier2=FakeTier2Adapter(),
            tier3=FakeTier3Adapter(),
            normalizer=PassthroughNormalizer(),
        )
