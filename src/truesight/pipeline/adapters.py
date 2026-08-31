from __future__ import annotations

import asyncio
import hashlib
import os
from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter
from typing import Any, Protocol

from .result_schema import (
    Tier1Result,
    Tier2Result,
    Tier3Result,
    default_forensics,
    default_provenance,
)


@dataclass
class Tier1Analysis:
    """Normalized hand-off from provenance analysis to orchestration."""

    summary: Tier1Result
    provenance: dict[str, Any] = field(default_factory=default_provenance)
    forensics: dict[str, Any] = field(default_factory=default_forensics)
    evidence: list[str] = field(default_factory=list)
    source: str | None = None


@dataclass(frozen=True)
class Tier2Input:
    """The deliberately limited context supplied to the VLM tier."""

    image_path: str
    forensic_overlay: str | None
    candidate_regions: list[dict[str, Any]]
    provenance_status: str | None
    provenance_verified: bool


class Tier1Adapter(Protocol):
    def analyze(self, original_image_path: str) -> Tier1Analysis:
        """Inspect the untouched uploaded image and return Tier 1 output."""


class Tier2Adapter(Protocol):
    def analyze(self, inputs: Tier2Input) -> Tier2Result:
        """Analyze the normalized image and concise supporting context."""


class Tier3Adapter(Protocol):
    def predict(self, normalized_image_path: str) -> Tier3Result:
        """Run the visual classifier on only the normalized RGB image."""


class ImageNormalizer(Protocol):
    def normalize(self, original_image_path: str) -> str:
        """Return a normalized derivative without modifying the original."""


class PassthroughNormalizer:
    """Development fallback used until image normalization is connected."""

    def normalize(self, original_image_path: str) -> str:
        return original_image_path


class RealTier1Adapter:
    """Connect Member 2's provenance/forensics implementation."""

    def __init__(
        self,
        output_dir: str | Path = "outputs/forensics",
        run_forensics: bool = True,
        run_openai_check: bool | None = None,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.run_forensics = run_forensics
        self.run_openai_check = (
            bool(os.getenv("OPENAI_API_KEY"))
            if run_openai_check is None
            else run_openai_check
        )

    def analyze(self, original_image_path: str) -> Tier1Analysis:
        from ..provenance.tier1 import analyze_tier1

        raw = analyze_tier1(
            original_image_path,
            output_dir=self.output_dir,
            run_forensics=self.run_forensics,
            run_openai_check=self.run_openai_check,
        )
        return Tier1Analysis(
            summary=Tier1Result(**raw["tier1"]),
            provenance=raw["provenance"],
            forensics=raw.get("forensics") or default_forensics(),
            evidence=list(raw.get("evidence", [])),
            source=raw.get("source"),
        )


class RealTier2Adapter:
    """Connect Member 3's available OpenAI/Gemini VLM providers."""

    def __init__(self, providers: list[Any] | None = None) -> None:
        self.providers = providers or []

    def analyze(self, inputs: Tier2Input) -> Tier2Result:
        if not self.providers:
            return Tier2Result(
                evidence=["VLM tier unavailable: no provider API key configured"],
            )

        from ..vlm import run_vlm_tier

        started = perf_counter()
        merged = asyncio.run(run_vlm_tier(inputs.image_path, self.providers))
        source = merged.source
        is_ai_generated = (
            None
            if source in (None, "Uncertain")
            else source != "Real"
        )
        return Tier2Result(
            is_ai_generated=is_ai_generated,
            confidence=merged.confidence,
            source=source,
            ai_coverage=merged.ai_coverage,
            evidence=merged.evidence,
            latency_ms=int((perf_counter() - started) * 1000),
        )


class ConvNeXtTier3Adapter:
    """Connect Member 1's checkpoint inference and optional Grad-CAM."""

    def __init__(
        self,
        checkpoint: str | Path | None,
        heatmap_dir: str | Path | None = "outputs/heatmaps",
        image_size: int = 224,
        device: str | None = None,
    ) -> None:
        self.checkpoint = Path(checkpoint) if checkpoint else None
        self.heatmap_dir = Path(heatmap_dir) if heatmap_dir else None
        self.image_size = image_size
        self.device = device
        self._model = None

    def _load_model(self):
        if self.checkpoint is None or not self.checkpoint.is_file():
            return None
        if self._model is None:
            from ..vision.inference import load_detector

            self._model = load_detector(self.checkpoint, device=self.device)
        return self._model

    def _generate_heatmap(self, model, image_path: str) -> str | None:
        if self.heatmap_dir is None:
            return None

        from ..vision.gradcam import (
            GradCAM,
            preprocess_image,
            save_heatmap_overlay,
        )

        device = next(model.parameters()).device
        tensor, display_image = preprocess_image(
            image_path,
            image_size=self.image_size,
        )
        tensor = tensor.to(device)

        cam = GradCAM(model)
        try:
            result = cam.generate(tensor)
            heatmap = result.heatmap
        finally:
            cam.remove_hooks()

        digest = hashlib.sha256(str(Path(image_path).resolve()).encode()).hexdigest()[:10]
        output_path = self.heatmap_dir / f"{Path(image_path).stem}_{digest}.png"
        save_heatmap_overlay(display_image, heatmap, output_path)
        return str(output_path)

    def predict(self, normalized_image_path: str) -> Tier3Result:
        started = perf_counter()
        model = self._load_model()
        if model is None:
            checkpoint = str(self.checkpoint) if self.checkpoint else "not configured"
            return Tier3Result(
                evidence=[f"ConvNeXt unavailable: checkpoint {checkpoint}"],
                latency_ms=int((perf_counter() - started) * 1000),
            )

        from ..vision.inference import predict_image

        probability = predict_image(
            model,
            normalized_image_path,
            image_size=self.image_size,
        )
        heatmap_path = self._generate_heatmap(model, normalized_image_path)
        return Tier3Result(
            probability=probability,
            heatmap_path=heatmap_path,
            evidence=["ConvNeXt visual classifier completed"],
            latency_ms=int((perf_counter() - started) * 1000),
        )


class FakeTier1Adapter:
    def analyze(self, original_image_path: str) -> Tier1Analysis:
        del original_image_path
        return Tier1Analysis(summary=Tier1Result())


class FakeTier2Adapter:
    def analyze(self, inputs: Tier2Input) -> Tier2Result:
        del inputs
        return Tier2Result(
            is_ai_generated=True,
            confidence=0.70,
            source="Stable Diffusion",
            ai_coverage=0.65,
            evidence=[
                "Synthetic-looking texture patterns were detected"
            ],
            latency_ms=0,
        )


class FakeTier3Adapter:
    def predict(self, normalized_image_path: str) -> Tier3Result:
        del normalized_image_path
        return Tier3Result(
            probability=0.82,
            heatmap_path=None,
            evidence=["ConvNeXt visual classifier completed"],
            latency_ms=0,
        )


@dataclass
class PipelineComponents:
    tier1: Tier1Adapter
    tier2: Tier2Adapter
    tier3: Tier3Adapter
    normalizer: ImageNormalizer

    @classmethod
    def fake(cls) -> PipelineComponents:
        return cls(
            tier1=FakeTier1Adapter(),
            tier2=FakeTier2Adapter(),
            tier3=FakeTier3Adapter(),
            normalizer=PassthroughNormalizer(),
        )

    @classmethod
    def real(
        cls,
        checkpoint: str | Path | None = None,
        run_vlm: bool = True,
        run_forensics: bool = True,
        generate_heatmap: bool = True,
    ) -> PipelineComponents:
        """Build the real Member 1-3 connections with optional dependencies."""
        try:
            from dotenv import load_dotenv

            load_dotenv()
        except ImportError:
            pass

        providers = []
        if run_vlm:
            if os.getenv("OPENAI_API_KEY"):
                from ..vlm import OpenAIProvider

                providers.append(OpenAIProvider())
            if os.getenv("GEMINI_API_KEY"):
                from ..vlm import GeminiProvider

                providers.append(GeminiProvider())

        project_root = Path(__file__).resolve().parents[3]
        resolved_checkpoint = checkpoint or os.getenv("TRUESIGHT_CHECKPOINT")
        if not resolved_checkpoint:
            resolved_checkpoint = (
                project_root / "outputs" / "member1" / "cifake_1000" / "best.pt"
            )

        return cls(
            tier1=RealTier1Adapter(
                output_dir=project_root / "outputs" / "forensics",
                run_forensics=run_forensics,
            ),
            tier2=RealTier2Adapter(providers),
            tier3=ConvNeXtTier3Adapter(
                checkpoint=resolved_checkpoint,
                heatmap_dir=(project_root / "outputs" / "heatmaps")
                if generate_heatmap
                else None,
            ),
            normalizer=PassthroughNormalizer(),
        )
