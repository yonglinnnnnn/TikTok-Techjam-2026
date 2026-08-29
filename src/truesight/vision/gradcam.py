from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch

from .model import ConvNeXtAIGCDetector


class ConvNeXtGradCAM:
    """Grad-CAM for the last ConvNeXt feature stage.

    The resulting map answers: "Which spatial regions most influenced the AIGC
    classifier logit?" It is not a ground-truth segmentation of generated pixels.
    """

    def __init__(self, model: ConvNeXtAIGCDetector, target_layer=None) -> None:
        self.model = model.eval()
        self.target_layer = target_layer or self.model.backbone.features[7][-1].block[0]
        self.activations: torch.Tensor | None = None
        self.gradients: torch.Tensor | None = None
        self._forward_handle = self.target_layer.register_forward_hook(self._save_activation)
        self._backward_handle = self.target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, _module, _inputs, output) -> None:
        self.activations = output.detach()

    def _save_gradient(self, _module, _grad_input, grad_output) -> None:
        self.gradients = grad_output[0].detach()

    def remove_hooks(self) -> None:
        self._forward_handle.remove()
        self._backward_handle.remove()

    def __call__(self, image_tensor: torch.Tensor) -> tuple[float, np.ndarray]:
        self.model.zero_grad(set_to_none=True)
        logits = self.model(image_tensor)
        target_logit = logits[0]
        target_logit.backward()

        if self.activations is None or self.gradients is None:
            raise RuntimeError("Grad-CAM hooks did not capture activations/gradients")

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1).clamp(min=0)
        cam = cam[0]
        cam = cam - cam.min()
        denominator = cam.max().clamp(min=1e-8)
        cam = cam / denominator
        probability = torch.sigmoid(target_logit).item()
        return probability, cam.cpu().numpy()


def overlay_heatmap(
    image_rgb: np.ndarray,
    cam: np.ndarray,
    output_path: str | Path,
    alpha: float = 0.45,
) -> None:
    h, w = image_rgb.shape[:2]
    heat = cv2.resize(cam, (w, h), interpolation=cv2.INTER_LINEAR)
    heat_u8 = np.uint8(np.clip(heat, 0.0, 1.0) * 255)
    heat_color = cv2.applyColorMap(heat_u8, cv2.COLORMAP_JET)
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    overlay = cv2.addWeighted(image_bgr, 1.0 - alpha, heat_color, alpha, 0.0)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), overlay)
