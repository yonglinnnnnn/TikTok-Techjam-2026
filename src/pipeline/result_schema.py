from dataclasses import asdict, dataclass
from typing import Any, Optional


# Tier 1: ProvenanceResult
@dataclass
class ProvenanceResult:
    detected: bool
    source: Optional[str]
    method: Optional[str]
    confidence: Optional[float]


# Tier 2: VLMResult
@dataclass
class VLMResult:
    source_estimate: Optional[str]
    confidence: Optional[float]
    explanation: Optional[str]


# Tier 3: ModelResult
@dataclass
class ModelResult:
    confidence: float
    is_ai_generated: bool
    heatmap_path: Optional[str] = None


# Tier 4: TrueSightResult
@dataclass
class TrueSightResult:
    image_path: str
    provenance: ProvenanceResult
    vlm: VLMResult
    model: ModelResult
    final_confidence: float
    final_is_ai_generated: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
