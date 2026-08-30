from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


def validate_probability(name: str, value: float | None) -> None:
    if value is not None and not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be between 0.0 and 1.0")

def default_severity_calibration() -> dict[str, Any]:
    return {
        "score": 0.0,
        "raw_score": 0.0,
        "calibrated": False,
        "method": "policy_v1",
        "model_version": None,
        "features": {
            "verified_ai_credential": 0.0,
            "provider_watermark": 0.0,
            "unverified_ai_credential": 0.0,
            "metadata_ai_marker": 0.0,
        },
        "error": None,
    }

def default_provenance() -> dict[str, Any]:
    return {
        "status": "no_checked",
        "conclusion": "Provenance checks have not completed",
        "signals": [],
        "file": {
            "sha256": None,
            "size_bytes": None,
            "format": None,
            "mime_type": None,
            "width": None,
            "height": None,
            "mode": None,
        },
        "c2pa": {
            "checked": False,
            "status": "not_checked",
            "present": None,
            "verified": False,
            "active_manifest": None,
            "validation_state": None,
            "history": None,
            "error": None,
        },
        "openai": {
            "checked": False,
            "status": "not_checked",
            "present": None,
            "verified": False,
            "source": None,
            "results": [],
            "error": None,
        },
        "metadata": {
            "checked": False,
            "present": None,
            "ai_markers": [],
            "error": None,
        },
    }

def default_forensics() -> dict[str, Any]:
    return {
        "integrity_risk_weight": 0.0,
        "candidate_coverage": 0.0,
        "candidate_regions": [],
        "feature_vector": {},
        "artifacts": {
            "candidate_mask": None,
            "vlm_overlay": None,
            "human_review": None,
        },
    }

# Checks metadata, verifies provenance, and detects watermarks
@dataclass
class Tier1Result:
    watermark_detected: bool | None = None
    provenance_verified: bool = False
    verified_ai_signal: bool = False
    verified_capture_signal: bool = False
    severity_weight: float = 0.0
    severity_calibration: dict[str, Any] = field(default_factory=default_severity_calibration)
    forensic_integrity_weight: float = 0.0
    requires_tier2: bool = True

    def __post_init__(self) -> None:
        validate_probability("severity_weight", self.severity_weight)
        validate_probability(
            "forensic_integrity_weight",
            self.forensic_integrity_weight,
        )

# VLM Output
@dataclass
class Tier2Result:
    is_ai_generated: bool | None = None
    confidence: float | None = None
    source: str | None = None
    ai_coverage: float | None = None
    evidence: list[str] = field(default_factory=list)
    latency_ms: int = 0

    def __post_init__(self) -> None:
        validate_probability("confidence", self.confidence)
        validate_probability("ai_coverage", self.ai_coverage)

# ConvNeXt Model Output
@dataclass
class Tier3Result:
    probability: float | None = None
    heatmap_path: str | None = None
    evidence: list[str] = field(default_factory=list)
    latency_ms: int = 0

    def __post_init__(self) -> None:
        validate_probability("probability", self.probability)

# Fusion Output - reads the outputs of all tiers and produces a final decision
@dataclass
class FusionResult:
    method: str
    calibrated: bool
    decision_threshold: float
    inputs: dict[str, Any] = field(default_factory=dict)
    latency_ms: int = 0

    def __post_init__(self) -> None:
        validate_probability(
            "decision_threshold",
            self.decision_threshold,
        )


# Complete TrueSight Result
@dataclass
class TrueSightResult:
    image_path: str
    schema_version: str = "1.0"
    is_ai_generated: bool | None = None
    confidence: float | None = None
    source: str | None = None
    ai_coverage: float | None = None
    heatmap_path: str | None = None
    provenance: dict[str, Any] = field(default_factory=default_provenance)
    forensics: dict[str, Any] = field(default_factory=default_forensics)
    evidence: list[str] = field(default_factory=list)
    latency_ms: int = 0
    tier1: Tier1Result = field(default_factory=Tier1Result)
    tier2: Tier2Result | None = None
    tier3: Tier3Result | None = None
    fusion: FusionResult | None = None

    def __post_init__(self) -> None:
        validate_probability("confidence", self.confidence)
        validate_probability("ai_coverage", self.ai_coverage)

    @classmethod
    def initialize(cls, image_path: str) -> TrueSightResult:
        return cls(image_path=image_path)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
