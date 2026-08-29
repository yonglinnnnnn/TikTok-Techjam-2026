from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torchvision.models import ConvNeXt_Tiny_Weights, convnext_tiny


@dataclass
class FreezeReport:
    trainable_parameters: int
    total_parameters: int
    trainable_ratio: float


class ConvNeXtAIGCDetector(nn.Module):
    """ConvNeXt-Tiny binary AIGC detector with staged transfer learning.

    Design principle:
      - keep ImageNet pretrained features initially frozen;
      - replace the 1000-way ImageNet classifier with a binary head;
      - selectively unfreeze the last N ConvNeXt stages;
      - use a conservative dropout head to reduce overfitting to dataset-specific
        generator artifacts.

    The model returns a single logit. sigmoid(logit) is the AIGC probability.
    """

    # Torchvision ConvNeXt-Tiny stage positions are 1, 3, 5, 7 in the features
    # Sequential. Keeping this mapping local makes the selective-unfreeze logic
    # explicit and easy to verify.
    STAGE_INDICES = (1, 3, 5, 7)

    def __init__(
        self,
        pretrained: bool = True,
        dropout: float = 0.15,
        unfreeze_stages: int = 2,
    ) -> None:
        super().__init__()

        weights = ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None
        backbone = convnext_tiny(weights=weights)
        in_features = backbone.classifier[-1].in_features
        backbone.classifier[-1] = nn.Linear(in_features, 1)
        backbone.classifier.insert(2, nn.Dropout(p=dropout))

        self.backbone = backbone
        self.unfreeze_stages = int(unfreeze_stages)
        self.freeze_all()
        self.unfreeze_last_stages(self.unfreeze_stages)
        self._unfreeze_classifier()

    def freeze_all(self) -> None:
        for parameter in self.backbone.parameters():
            parameter.requires_grad = False

    def _unfreeze_classifier(self) -> None:
        for parameter in self.backbone.classifier.parameters():
            parameter.requires_grad = True

    def unfreeze_last_stages(self, n: int) -> None:
        if n < 0 or n > len(self.STAGE_INDICES):
            raise ValueError(f"unfreeze_stages must be in [0, {len(self.STAGE_INDICES)}]")
        for stage_index in self.STAGE_INDICES[-n:] if n else ():
            for parameter in self.backbone.features[stage_index].parameters():
                parameter.requires_grad = True

        # Unfreezing a stage without its preceding downsampling layer is usually
        # sufficient for transfer learning. Keep transition layers frozen to
        # limit the number of trainable parameters and reduce catastrophic drift.

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x).squeeze(-1)

    def logits(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x)

    @torch.no_grad()
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.forward(x))

    def freeze_report(self) -> FreezeReport:
        total = sum(parameter.numel() for parameter in self.parameters())
        trainable = sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)
        return FreezeReport(
            trainable_parameters=trainable,
            total_parameters=total,
            trainable_ratio=trainable / max(total, 1),
        )

    @classmethod
    def from_checkpoint(cls, checkpoint: dict, map_location: str | torch.device = "cpu") -> "ConvNeXtAIGCDetector":
        model_cfg = checkpoint.get("model_config", {})
        model = cls(
            pretrained=False,
            dropout=model_cfg.get("dropout", 0.15),
            unfreeze_stages=model_cfg.get("unfreeze_stages", 2),
        )
        model.load_state_dict(checkpoint["model_state"], strict=True)
        model.to(map_location)
        return model
