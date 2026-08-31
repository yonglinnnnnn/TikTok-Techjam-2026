#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import sys
import uuid
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from truesight.pipeline import PipelineComponents, TrueSightResult, run_pipeline

from components import render_result

ALLOWED_UPLOAD_TYPES = ["jpg", "jpeg", "png", "webp"]
MAX_UPLOAD_BYTES = 20 * 1024 * 1024
RESULT_KEY = "truesight_result"
UPLOAD_ID_KEY = "truesight_upload_id"
UPLOAD_TEMP_ROOT = PROJECT_ROOT / ".truesight-tmp"


@st.cache_resource
def pipeline_components() -> PipelineComponents:
    return PipelineComponents.real()


def analyze_upload(file_name: str, file_bytes: bytes) -> TrueSightResult:
    suffix = Path(file_name).suffix.lower()
    if suffix not in {f".{extension}" for extension in ALLOWED_UPLOAD_TYPES}:
        raise ValueError(f"Unsupported image format: {suffix or 'missing'}")

    UPLOAD_TEMP_ROOT.mkdir(parents=True, exist_ok=True)
    image_path = UPLOAD_TEMP_ROOT / f"{uuid.uuid4().hex}{suffix}"
    image_path.write_bytes(file_bytes)
    try:
        return run_pipeline(str(image_path), components=pipeline_components())
    finally:
        image_path.unlink(missing_ok=True)


def main() -> None:
    st.set_page_config(
        page_title="TrueSight",
        page_icon="🔎",
        layout="wide",
    )

    st.title("TrueSight")
    st.write(
        "Inspect image provenance, visual signals, and model evidence through "
        "one explainable detection pipeline."
    )
    st.warning(
        "Prototype mode: Members 1-3 are connected, but final score fusion is "
        "still uncalibrated. Missing API keys are reported as unavailable."
    )

    uploaded_file = st.file_uploader(
        "Upload an image",
        type=ALLOWED_UPLOAD_TYPES,
        accept_multiple_files=False,
        help="Supported formats: JPEG, PNG, and WebP. Maximum size: 20 MB.",
    )

    if uploaded_file is None:
        st.info("Upload an image to begin analysis.")
        return

    file_bytes = uploaded_file.getvalue()
    if len(file_bytes) > MAX_UPLOAD_BYTES:
        st.error("The uploaded image exceeds the 20 MB demo limit.")
        return

    upload_id = hashlib.sha256(file_bytes).hexdigest()
    if st.session_state.get(UPLOAD_ID_KEY) != upload_id:
        st.session_state.pop(RESULT_KEY, None)

    preview, action = st.columns([2, 1])
    with preview:
        st.image(file_bytes, caption=uploaded_file.name, width="stretch")
    with action:
        st.write(f"**File:** {uploaded_file.name}")
        st.write(f"**Size:** {len(file_bytes) / 1024:.1f} KB")
        analyze_clicked = st.button(
            "Analyze image",
            type="primary",
            width="stretch",
        )

    if analyze_clicked:
        try:
            with st.spinner("Running TrueSight analysis..."):
                result = analyze_upload(uploaded_file.name, file_bytes)
        except (FileNotFoundError, ValueError) as exc:
            st.error(f"Analysis failed: {exc}")
            st.session_state.pop(RESULT_KEY, None)
        except Exception as exc:
            print(exc)
            st.error("Analysis failed because an internal component returned an error.")
            st.session_state.pop(RESULT_KEY, None)
        else:
            st.session_state[RESULT_KEY] = result
            st.session_state[UPLOAD_ID_KEY] = upload_id

    stored_result = st.session_state.get(RESULT_KEY)
    if (
        isinstance(stored_result, TrueSightResult)
        and st.session_state.get(UPLOAD_ID_KEY) == upload_id
    ):
        st.divider()
        render_result(stored_result)


if __name__ == "__main__":
    main()
