# Blind forensics pipeline

This pipeline analyzes one image without receiving its original. It produces
candidate manipulation maps from four complementary signals:

- JPEG recompression/error-level differences;
- local high-pass/noise-energy inconsistency;
- local 8x8 JPEG-grid inconsistency;
- spatially separated ORB matches for possible copy-move operations.

Install and run from the repository root:

```powershell
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
cd src
python -m truesight.provenance.forensics ..\data\image4.jpg
```

Or run the combined Tier 1 pipeline:

```powershell
cd src
python -m truesight.provenance.tier1 ..\data\image4.jpg
```

No model is loaded. The default artifacts are intentionally small:

- `vlm_overlay`: the only forensic image normally sent to Tier 2, together with
  the unmodified input image;
- `candidate_mask`: a machine-readable localization aid, not an AI-coverage
  estimate;
- `human_review`: a three-panel visualization for developers and judges. It is
  not sent to a model.

The compact `feature_vector`, `integrity_risk_weight`, candidate coverage and
candidate boxes are suitable for routing or calibrated score fusion. Tier 3's
ConvNeXt still receives the original image, not these handcrafted maps, unless
it is deliberately retrained as a multi-input model.

For manual checking, open `artifacts.human_review`. It contains
three side-by-side views: numbered candidate boxes, the binary red-mask overlay,
and the continuous anomaly heatmap. Region numbers correspond to the IDs in
`candidate_regions`.

Each candidate includes a human-readable ID and location, an inclusive pixel box,
a normalized box for resolution-independent consumers, anomaly strength,
and multi-signal support. For example, `R1` in JSON is the box marked `R1` in
the review image.

The capped `integrity_risk_weight` is deliberately not
returned as top-level confidence. Scene texture, camera processing and social-
media compression can create the same traces. The deterministic stage therefore
always recommends continuing to Tier 2 and Tier 3 unless verified provenance has
already triggered the fast path.

Recommended Tier 2 input consists of the original image, `vlm_overlay`, candidate
regions and, if useful, the feature vector. Ask the VLM to
explain whether highlighted regions correspond to plausible content, exposure
boundaries, repeated natural texture, or actual manipulation. Tier 3 remains
responsible for the learned classification and Grad-CAM result.

Use `--debug-artifacts` (or `--debug-forensics` on the combined Tier 1 command)
to expose the raw ELA, noise, JPEG-grid, copy-move and combined maps, their raw
signals, threshold, timing and region crop sheet. These diagnostics are for
tuning and failure analysis; they are not normal downstream inputs.
