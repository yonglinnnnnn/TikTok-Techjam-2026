from __future__ import annotations

from dataclasses import dataclass

from .result_schema import Tier1Result


@dataclass(frozen=True)
class RoutingDecision:
    """Execution plan selected from the normalized Tier 1 summary."""

    run_tier2: bool
    run_tier3: bool
    fast_path: bool
    reason: str


def decide_route(tier1: Tier1Result) -> RoutingDecision:
    """Choose the safe route after provenance analysis.

    Only a verified AI signal can skip downstream analysis. A watermark that
    was merely detected, verified capture provenance, missing metadata, and
    provider failures are all inconclusive and must continue downstream.
    """

    if tier1.verified_ai_signal:
        return RoutingDecision(
            run_tier2=False,
            run_tier3=False,
            fast_path=True,
            reason="verified_ai_signal",
        )

    return RoutingDecision(
        run_tier2=True,
        run_tier3=True,
        fast_path=False,
        reason="no_verified_ai_signal",
    )
