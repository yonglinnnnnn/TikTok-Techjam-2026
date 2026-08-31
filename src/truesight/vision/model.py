from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torchvision.models import (
    ConvNeXt_Tiny_Weights,
    convnext_tiny,
)


@dataclass
class FreezeReport:
    """Summary of frozen and trainable model parameters."""

    trainable_parameters: int
    total_parameters: int
    trainable_ratio: float


class ConvNeXtAIGCDetector(nn.Module):
    """
    ConvNeXt-Tiny binary AI-generated-image detector.

    The model returns one raw logit.

    Convert the logit into an AI probability with:

        probability = torch.sigmoid(logit)

    Labels:

        0 = real
        1 = AI-generated or tampered
    """

    # TorchVision ConvNeXt-Tiny layout:
    #
    # features[0] = stem
    # features[1] = stage 1
    # features[2] = downsampling layer
    # features[3] = stage 2
    # features[4] = downsampling layer
    # features[5] = stage 3
    # features[6] = downsampling layer
    # features[7] = stage 4
    #
    # Therefore, actual ConvNeXt stages are at:
    #
    #     1, 3, 5, 7
    #
    STAGE_INDICES = (1, 3, 5, 7)

    def __init__(
        self,
        pretrained: bool = True,
        dropout: float = 0.15,
        unfreeze_stages: int = 2,
    ) -> None:
        super().__init__()

        self.unfreeze_stages = int(
            unfreeze_stages
        )

        if self.unfreeze_stages < 0:
            raise ValueError(
                "unfreeze_stages cannot be negative"
            )

        if self.unfreeze_stages > len(
            self.STAGE_INDICES
        ):
            raise ValueError(
                "unfreeze_stages must be between "
                f"0 and {len(self.STAGE_INDICES)}"
            )

        weights = (
            ConvNeXt_Tiny_Weights.DEFAULT
            if pretrained
            else None
        )

        self.backbone = convnext_tiny(
            weights=weights
        )

        # Replace the original ImageNet 1000-class
        # classifier with a binary classifier.
        in_features = (
            self.backbone.classifier[-1]
            .in_features
        )

        self.backbone.classifier[-1] = nn.Linear(
            in_features,
            1,
        )

        # The original classifier normally contains:
        #
        #     LayerNorm
        #     Flatten
        #     Linear
        #
        # Insert dropout before the final Linear layer.
        self.backbone.classifier.insert(
            2,
            nn.Dropout(p=dropout),
        )

        # Freeze everything first.
        self.freeze_all()

        # Unfreeze the requested final stages.
        self.unfreeze_last_stages(
            self.unfreeze_stages
        )

        # The classifier is always trainable.
        self._unfreeze_classifier()

    def freeze_all(self) -> None:
        """Freeze every parameter in the ConvNeXt model."""
        for parameter in self.backbone.parameters():
            parameter.requires_grad = False

    def _unfreeze_classifier(self) -> None:
        """Make the replacement binary classifier trainable."""
        for parameter in (
            self.backbone.classifier.parameters()
        ):
            parameter.requires_grad = True

    def unfreeze_last_stages(self, n: int) -> None:
        """
        Unfreeze the final n ConvNeXt stages.

        Examples:

            n = 0:
                classifier only

            n = 1:
                final ConvNeXt stage + classifier

            n = 2:
                final two ConvNeXt stages + classifier

            n = 3:
                final three ConvNeXt stages + classifier
        """
        if n < 0 or n > len(
            self.STAGE_INDICES
        ):
            raise ValueError(
                "unfreeze_stages must be between "
                f"0 and {len(self.STAGE_INDICES)}"
            )

        selected_indices = (
            self.STAGE_INDICES[-n:]
            if n > 0
            else ()
        )

        for stage_index in selected_indices:
            stage = self.backbone.features[
                stage_index
            ]

            stage_parameters = list(
                stage.parameters()
            )

            if not stage_parameters:
                raise RuntimeError(
                    "Selected ConvNeXt stage has no "
                    f"parameters: features[{stage_index}]"
                )

            for parameter in stage_parameters:
                parameter.requires_grad = True

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """
        Run a forward pass.

        Returns:
            Tensor of shape [batch_size] containing
            one raw logit per image.
        """
        return self.backbone(x).squeeze(-1)

    def logits(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """Alias for forward()."""
        return self.forward(x)

    @torch.no_grad()
    def predict_proba(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """
        Return AI probabilities.

        Values close to 0 indicate real.
        Values close to 1 indicate AI-generated.
        """
        return torch.sigmoid(
            self.forward(x)
        )

    def freeze_report(self) -> FreezeReport:
        """Return trainable and total parameter counts."""
        total = sum(
            parameter.numel()
            for parameter in self.parameters()
        )

        trainable = sum(
            parameter.numel()
            for parameter in self.parameters()
            if parameter.requires_grad
        )

        return FreezeReport(
            trainable_parameters=trainable,
            total_parameters=total,
            trainable_ratio=(
                trainable / max(total, 1)
            ),
        )

    def trainable_parameter_names(
        self,
    ) -> list[str]:
        """Return names of all trainable parameters."""
        return [
            name
            for name, parameter in self.named_parameters()
            if parameter.requires_grad
        ]

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint: dict,
        map_location: str | torch.device = "cpu",
    ) -> "ConvNeXtAIGCDetector":
        """
        Reconstruct a detector from a saved checkpoint.
        """
        model_config = checkpoint.get(
            "model_config",
            {},
        )

        model = cls(
            pretrained=False,
            dropout=model_config.get(
                "dropout",
                0.15,
            ),
            unfreeze_stages=model_config.get(
                "unfreeze_stages",
                2,
            ),
        )

        model.load_state_dict(
            checkpoint["model_state"],
            strict=True,
        )

        model.to(map_location)

        return model

    