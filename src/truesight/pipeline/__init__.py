"""TrueSight image-analysis pipeline."""

from .adapters import (
    FakeTier1Adapter,
    FakeTier2Adapter,
    FakeTier3Adapter,
    ImageNormalizer,
    PassthroughNormalizer,
    PipelineComponents,
    Tier1Adapter,
    Tier1Analysis,
    Tier2Adapter,
    Tier2Input,
    Tier3Adapter,
)
from .orchestrator import run_pipeline
from .router import RoutingDecision, decide_route
from .result_schema import (
    FusionResult,
    Tier1Result,
    Tier2Result,
    Tier3Result,
    TrueSightResult,
)

__all__ = [
    "FakeTier1Adapter",
    "FakeTier2Adapter",
    "FakeTier3Adapter",
    "FusionResult",
    "ImageNormalizer",
    "PassthroughNormalizer",
    "PipelineComponents",
    "RoutingDecision",
    "Tier1Adapter",
    "Tier1Analysis",
    "Tier1Result",
    "Tier2Adapter",
    "Tier2Input",
    "Tier2Result",
    "Tier3Adapter",
    "Tier3Result",
    "TrueSightResult",
    "decide_route",
    "run_pipeline",
]
