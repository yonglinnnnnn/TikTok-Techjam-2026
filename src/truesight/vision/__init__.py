"""Member 1 computer-vision components.

Imports are intentionally kept lightweight so model-only tooling does not require
Albumentations at import time. Training/data modules import their dependencies
when used.
"""

from .model import ConvNeXtAIGCDetector

__all__ = ["ConvNeXtAIGCDetector"]
