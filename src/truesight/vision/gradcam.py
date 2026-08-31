from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageOps
from torchvision import transforms


IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
    ".bmp",
}


@dataclass
class GradCAMOutput:
    """Grad-CAM result for one image."""

    heatmap: np.ndarray
    logit: float
    probability: float


def get_device(
    requested_device: str | None = None,
) -> torch.device:
    """
    Select the inference device.

    If requested_device is None, CUDA is used when available.
    """
    if requested_device:
        return torch.device(requested_device)

    return torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )


def get_default_target_layer(model):
    """
    Return the final ConvNeXt feature stage.

    TorchVision ConvNeXt-Tiny has its final feature stage at:

        model.backbone.features[7]

    This layer produces spatial feature maps suitable for Grad-CAM.
    """
    try:
        return model.backbone.features[7]
    except AttributeError as error:
        raise AttributeError(
            "The supplied model does not contain "
            "model.backbone.features[7]."
        ) from error


def preprocess_image(
    image_path: str | Path,
    image_size: int = 224,
) -> tuple[torch.Tensor, Image.Image]:
    """
    Load and preprocess an image.

    Returns:

        model_input:
            Tensor with shape [1, 3, image_size, image_size]

        display_image:
            Resized PIL image used for the heatmap overlay
    """
    image_path = Path(image_path)

    image = Image.open(image_path).convert("RGB")

    # ImageOps.fit matches the image to the square model input.
    # This ensures the heatmap aligns with the displayed image.
    display_image = ImageOps.fit(
        image,
        (image_size, image_size),
        method=Image.Resampling.BICUBIC,
    )

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[
                    0.485,
                    0.456,
                    0.406,
                ],
                std=[
                    0.229,
                    0.224,
                    0.225,
                ],
            ),
        ]
    )

    tensor = transform(display_image)

    return tensor.unsqueeze(0), display_image


class GradCAM:
    """
    Grad-CAM implementation for ConvNeXt.

    Grad-CAM uses:

        feature activations
        +
        gradients of the prediction

    to estimate which spatial regions influenced
    the model's decision.
    """

    def __init__(
        self,
        model,
        target_layer=None,
    ) -> None:
        self.model = model
        self.model.eval()

        self.target_layer = (
            target_layer
            if target_layer is not None
            else get_default_target_layer(model)
        )

        self.activations: torch.Tensor | None = None
        self.gradients: torch.Tensor | None = None

        self.forward_handle = (
            self.target_layer.register_forward_hook(
                self._forward_hook
            )
        )

        self.backward_handle = (
            self.target_layer.register_full_backward_hook(
                self._backward_hook
            )
        )

    def _forward_hook(
        self,
        module,
        inputs,
        output,
    ) -> None:
        """Store feature activations."""
        self.activations = output

    def _backward_hook(
        self,
        module,
        grad_inputs,
        grad_outputs,
    ) -> None:
        """Store gradients flowing through the target layer."""
        self.gradients = grad_outputs[0]

    def remove_hooks(self) -> None:
        """Remove PyTorch hooks."""
        self.forward_handle.remove()
        self.backward_handle.remove()

    def generate(
        self,
        model_input: torch.Tensor,
    ) -> GradCAMOutput:
        """
        Generate a Grad-CAM heatmap.

        The model logit is used as the target because a larger
        logit corresponds to a stronger AI-generated prediction.
        """
        self.model.eval()
        self.model.zero_grad(set_to_none=True)

        self.activations = None
        self.gradients = None

        logits = self.model(model_input)
        logit = logits.reshape(-1)[0]

        # Grad-CAM requires gradients, so do not use torch.no_grad().
        logit.backward()

        if self.activations is None:
            raise RuntimeError(
                "Grad-CAM did not capture activations."
            )

        if self.gradients is None:
            raise RuntimeError(
                "Grad-CAM did not capture gradients."
            )

        activations = self.activations
        gradients = self.gradients

        # Average gradients across height and width.
        weights = gradients.mean(
            dim=(2, 3),
            keepdim=True,
        )

        # Weighted sum of feature channels.
        cam = (
            weights * activations
        ).sum(dim=1, keepdim=True)

        # Keep only positive evidence.
        cam = F.relu(cam)

        # Resize the heatmap to the model input size.
        cam = F.interpolate(
            cam,
            size=(
                model_input.shape[-2],
                model_input.shape[-1],
            ),
            mode="bilinear",
            align_corners=False,
        )

        heatmap = cam[0, 0].detach().cpu().numpy()

        # Normalize to [0, 1].
        heatmap -= heatmap.min()

        maximum = heatmap.max()

        if maximum > 1e-8:
            heatmap /= maximum

        probability = float(
            torch.sigmoid(logit).detach().cpu().item()
        )

        return GradCAMOutput(
            heatmap=heatmap,
            logit=float(
                logit.detach().cpu().item()
            ),
            probability=probability,
        )


def save_heatmap_overlay(
    display_image: Image.Image | np.ndarray,
    heatmap: np.ndarray,
    output_path: str | Path,
    alpha: float = 0.45,
) -> None:
    """
    Save an image with a Grad-CAM overlay.

    Red/yellow areas indicate stronger positive influence
    on the AI-generated prediction.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if isinstance(display_image, Image.Image):
        image_array = np.asarray(
            display_image.convert("RGB")
        )
    else:
        image_array = np.asarray(display_image)
        if image_array.ndim != 3 or image_array.shape[2] != 3:
            raise ValueError(
                "display_image must be an RGB image with shape [H, W, 3]."
            )
        if image_array.dtype != np.uint8:
            image_array = np.clip(image_array, 0, 255).astype(np.uint8)

    heatmap_uint8 = np.uint8(
        np.clip(heatmap, 0.0, 1.0) * 255
    )

    colored_heatmap = cv2.applyColorMap(
        heatmap_uint8,
        cv2.COLORMAP_JET,
    )

    colored_heatmap = cv2.cvtColor(
        colored_heatmap,
        cv2.COLOR_BGR2RGB,
    )

    overlay = cv2.addWeighted(
        image_array,
        1.0 - alpha,
        colored_heatmap,
        alpha,
        0.0,
    )

    Image.fromarray(overlay).save(
        output_path
    )


def generate_gradcam_for_image(
    model,
    image_path: str | Path,
    output_path: str | Path,
    image_size: int = 224,
    device: str | torch.device | None = None,
) -> GradCAMOutput:
    """
    Generate and save a Grad-CAM overlay for one image.
    """
    selected_device = get_device(
        str(device)
        if device is not None
        else None
    )

    model = model.to(selected_device)
    model.eval()

    model_input, display_image = preprocess_image(
        image_path,
        image_size=image_size,
    )

    model_input = model_input.to(
        selected_device
    )

    gradcam = GradCAM(model)

    try:
        result = gradcam.generate(
            model_input
        )

        save_heatmap_overlay(
            display_image=display_image,
            heatmap=result.heatmap,
            output_path=output_path,
        )

    finally:
        gradcam.remove_hooks()

    return result
