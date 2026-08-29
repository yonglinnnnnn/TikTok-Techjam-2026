from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from truesight.vision.model import ConvNeXtAIGCDetector


def test_model_output_shape():
    model = ConvNeXtAIGCDetector(pretrained=False, unfreeze_stages=1)
    x = torch.randn(2, 3, 224, 224)
    logits = model(x)
    assert logits.shape == (2,)


def test_selective_unfreeze():
    model = ConvNeXtAIGCDetector(pretrained=False, unfreeze_stages=1)
    trainable = [name for name, p in model.named_parameters() if p.requires_grad]
    assert any(name.startswith("backbone.classifier") for name in trainable)
    assert any(name.startswith("backbone.features.7") for name in trainable)
    assert not any(name.startswith("backbone.features.1") for name in trainable)
