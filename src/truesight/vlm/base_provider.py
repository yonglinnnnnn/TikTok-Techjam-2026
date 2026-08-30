from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class VLMResult:
    provider: str
    source_estimate: str | None = None
    ai_coverage: float | None = None
    confidence: float | None = None
    reasoning: str | None = None
    evidence: list[str] = field(default_factory=list)
    raw_response: dict[str, Any] | None = None
    latency_ms: int | None = None
    error: str | None = None

    def is_valid(self) -> bool:
        return self.error is None and self.source_estimate is not None


class BaseVLMProvider(ABC):
    name: str = "base"

    @abstractmethod
    async def _call(self, image_path: str, prompt: str) -> dict[str, Any]:
        """Make the API call. Return raw parsed JSON. Raise on failure."""
        raise NotImplementedError

    async def analyze(self, image_path: str) -> VLMResult:
        from .prompts import CONSISTENCY_CHECK_PROMPT
        from .parser import parse_vlm_response

        start = time.monotonic()
        try:
            raw = await self._call(image_path, CONSISTENCY_CHECK_PROMPT)
            result = parse_vlm_response(raw, provider=self.name)
            result.latency_ms = int((time.monotonic() - start) * 1000)
            return result
        except Exception as exc:  # noqa: BLE001
            return VLMResult(
                provider=self.name,
                error=f"{type(exc).__name__}: {exc}",
                latency_ms=int((time.monotonic() - start) * 1000),
            )