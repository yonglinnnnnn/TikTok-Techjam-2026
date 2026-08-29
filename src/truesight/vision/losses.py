from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class BinaryAIGCLoss(nn.Module):
    """BCE-with-logits plus optional clean/transformed consistency loss."""

    def __init__(self, consistency_weight: float = 0.05, label_smoothing: float = 0.0) -> None:
        super().__init__()
        self.consistency_weight = float(consistency_weight)
        self.label_smoothing = float(label_smoothing)

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        consistency_logits: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        smoothed_targets = targets
        if self.label_smoothing > 0:
            smoothed_targets = targets * (1.0 - self.label_smoothing) + 0.5 * self.label_smoothing

        supervised = F.binary_cross_entropy_with_logits(logits, smoothed_targets)
        total = supervised
        consistency = torch.zeros((), device=logits.device)

        if consistency_logits is not None and self.consistency_weight > 0:
            # Symmetric probability consistency is intentionally simple and stable.
            p1 = torch.sigmoid(logits.detach())
            p2 = torch.sigmoid(consistency_logits)
            consistency = F.mse_loss(p2, p1)
            total = total + self.consistency_weight * consistency

        return total, {
            "supervised": float(supervised.detach().item()),
            "consistency": float(consistency.detach().item()),
            "total": float(total.detach().item()),
        }
