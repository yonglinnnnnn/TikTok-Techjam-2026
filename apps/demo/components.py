from __future__ import annotations

import json
from pathlib import Path

import streamlit as st

from truesight.pipeline import TrueSightResult


def format_probability(value: float | None) -> str:
    return "Unknown" if value is None else f"{value:.1%}"


def render_summary(result: TrueSightResult) -> None:
    st.subheader("Detection result")

    if result.is_ai_generated is True:
        st.error("Likely AI-generated")
    elif result.is_ai_generated is False:
        st.success("Likely not AI-generated")
    else:
        st.warning("Inconclusive")

    confidence = result.confidence or 0.0
    st.progress(
        confidence,
        text=f"Final confidence: {format_probability(result.confidence)}",
    )

    source, coverage, latency = st.columns(3)
    source.metric("Source estimate", result.source or "Unknown")
    coverage.metric("Estimated AI coverage", format_probability(result.ai_coverage))
    latency.metric("Pipeline latency", f"{result.latency_ms} ms")

    if result.fusion is not None:
        calibration = "calibrated" if result.fusion.calibrated else "uncalibrated"
        st.caption(
            f"Fusion: {result.fusion.method} · {calibration} · "
            f"threshold {result.fusion.decision_threshold:.2f}"
        )


def render_provenance(result: TrueSightResult) -> None:
    st.subheader("Provenance")

    status, verified, watermark = st.columns(3)
    status.metric(
        "Status",
        str(result.provenance.get("status", "unknown")).replace("_", " ").title(),
    )
    verified.metric(
        "Verified AI signal",
        "Yes" if result.tier1.verified_ai_signal else "No",
    )
    watermark_value = result.tier1.watermark_detected
    watermark.metric(
        "Watermark",
        "Unknown" if watermark_value is None else ("Detected" if watermark_value else "None"),
    )

    conclusion = result.provenance.get("conclusion")
    if conclusion:
        st.write(conclusion)

    with st.expander("Provenance details"):
        st.json(result.provenance, expanded=2)


def render_evidence(result: TrueSightResult) -> None:
    st.subheader("Evidence")

    if not result.evidence:
        st.info("No supporting evidence was returned.")
        return

    for item in result.evidence:
        st.markdown(f"- {item}")


def render_visual_evidence(result: TrueSightResult) -> None:
    artifacts = result.forensics.get("artifacts", {})
    visual_paths = [
        ("Forensic review", artifacts.get("human_review")),
        ("Grad-CAM explanation", result.heatmap_path),
    ]
    available = [
        (label, Path(path))
        for label, path in visual_paths
        if path and Path(path).is_file()
    ]

    st.subheader("Visual evidence")
    if not available:
        st.info("No forensic review panel or Grad-CAM heatmap was generated.")
        return

    columns = st.columns(len(available))
    for column, (label, path) in zip(columns, available, strict=True):
        with column:
            st.image(path, caption=label, width="stretch")


def render_raw_result(result: TrueSightResult) -> None:
    result_dict = result.to_dict()
    json_result = json.dumps(result_dict, indent=2, ensure_ascii=False)

    with st.expander("Raw JSON result"):
        st.json(result_dict, expanded=2)

    st.download_button(
        "Download JSON result",
        data=f"{json_result}\n",
        file_name="truesight_prediction.json",
        mime="application/json",
    )


def render_result(result: TrueSightResult) -> None:
    render_summary(result)
    st.divider()
    render_provenance(result)
    st.divider()
    render_evidence(result)
    st.divider()
    render_visual_evidence(result)
    st.divider()
    render_raw_result(result)
