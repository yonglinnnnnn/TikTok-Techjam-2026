from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class Member1Prediction:
    """Stable hand-off object from Member 1 to the rest of TrueSight.

    The fields owned by Members 2 and 3 are intentionally represented as None.
    Member 1 must not make decisions from those fields.
    """

    image_path: str
    pred: float
    heatmap_path: str | None = None
    provenance: dict[str, Any] | None = None
    vlm: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "image_path": self.image_path,
            "pred": self.pred,
            "heatmap_path": self.heatmap_path,
            "provenance": self.provenance,
            "vlm": self.vlm,
        }


# Future integration boundary only. Member 4 owns the final score-fusion policy.
def combine_member_outputs(member1: Member1Prediction, **_future_outputs: Any) -> Member1Prediction:
    """Return Member 1's prediction unchanged.

    This placeholder exists solely to reserve a clean interface. Do not add
    provenance/VLM fusion logic here; that belongs to the integration layer.
    """
    return member1
