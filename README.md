# TrueSight

TrueSight is a three-tier pipeline for detecting AI-generated and AI-modified
images under real-world transformations. It combines cryptographically
verifiable provenance, semantic visual reasoning, and a learned visual
classifier. The tiers deliberately use different evidence, so a weakness in one
method does not become the system's single point of failure.

## Architecture

```text
                         untouched uploaded bytes
                                      |
                                      v
          +------------------------------------------------+
          | Tier 1: provenance and blind forensics        |
          | C2PA | provider check | metadata | local maps |
          +------------------------------------------------+
                    | verified AI credential/watermark
                    +------------------------> fast-path verdict
                    |
                    | otherwise
                    v
       normalized derivative (the original remains untouched)
                    |                         |
                    v                         v
       Tier 2: VLM semantic analysis   Tier 3: ConvNeXt-Tiny
       source / coverage / evidence    P(AIGC) + Grad-CAM
                    \                         /
                     \                       /
                      v                     v
                    calibrated fusion and UI result
```

The original uploaded file must never be re-encoded before Tier 1. Resizing,
RGB conversion, EXIF orientation, and recompression can remove or invalidate
provenance evidence. Tier 2 and Tier 3 use a separate normalized derivative;
the original is retained for provenance checks and audit.

## Result contract and ownership

`schemas/prediction.schema.json` is the canonical result contract. Tier 1
creates the complete envelope with `tier2`, `tier3`, and `fusion` set to `null`.
Each tier writes only its own block. Fusion writes the final top-level verdict.

| Component | Produces | Does not do |
|---|---|---|
| Tier 1 | Provenance, forensic artifacts, routing signals | Re-encode the source or infer authenticity from missing metadata |
| Tier 2 | VLM verdict, source estimate, AI coverage, semantic evidence | Treat absent provenance as proof of a real image |
| Tier 3 | ConvNeXt probability and Grad-CAM heatmap | Use provenance or forensic maps unless trained as a multi-input model |
| Fusion | Final verdict, confidence, threshold, calibration record | Add correlated scores directly |
| UI | Human-readable evidence and review artifacts | Show raw debug data by default |

Use `null` for unknown, unavailable, or not-yet-produced values. Use `false`
only after a check actually ran and returned a negative result. Missing C2PA,
for example, is not evidence that an image is authentic.

## Tier 1: provenance and blind forensics

Tier 1 has two separate roles:

1. Verify attributable provenance evidence in the untouched file.
2. Produce weak local anomaly evidence without requiring an original/reference
   image.

Only verified AI-attributed provenance can skip the later tiers. Verified camera
capture, missing provenance, invalid credentials, unverified marks, and forensic
anomalies remain inconclusive and continue downstream.

### C2PA content credentials

[C2PA](https://c2pa.org/) content credentials are signed provenance records
embedded in an asset. TrueSight reads the manifest store and returns:

- the active/root manifest and reachable ingredient manifests;
- ingredient relationships, missing references, cycles, and unsigned ingredients;
- declared actions from oldest to active asset;
- origin/digital-source type, declared edits, transformations, and metadata
  edits; and
- validation state and signer trust.

The checker separates a credential that merely parses from a credential trusted
by policy. Only the C2PA SDK's `Trusted` validation state is considered verified;
a `Valid` but untrusted signer is not sufficient. A C2PA history tells us what
the signed history declares. `content_edited: false` does not rule out an
undeclared edit or a provenance history stripped by a platform.

The auditable history lives in `provenance.c2pa.history`: `chain` represents the
manifest graph and `timeline` presents signed declared actions in chronological
order.

### Provider provenance, watermarks, and metadata

With `OPENAI_API_KEY` configured, Tier 1 sends the untouched bytes to the
OpenAI content-provenance check. It can report supported OpenAI C2PA and image
SynthID signals. Every source is normalized to a common signal with `kind`,
`provider`, `status`, `present`, `verified`, `ai_claim`, optional source,
verification basis, and evidence score.

Important states are:

- `unavailable`: no API key or provider dependency is available;
- `not_checked`: the check was explicitly skipped;
- `not_detected`: the completed provider check found no supported signal; and
- `verified`: a verified AI-attributed credential or provider-confirmed
  watermark.

Embedded metadata, such as a `Software` tag, is only weak evidence because it
can be forged, edited, or stripped. The project does not use a separate
key-dependent watermark provider or the removed `synthid-text` package.

### Severity and calibration

The default provenance policy avoids double-counting correlated local and
provider checks of the same credential. It uses the strongest independent
signal:

| Signal | Default contribution |
|---|---:|
| Verified AI content credential | 0.98 |
| Provider-confirmed AI watermark | 0.97 |
| Present but unverified AI credential | 0.35 |
| AI-related embedded metadata | 0.15 |

These are evidence contributions, not calibrated real-world probabilities. Use
`scripts/calibrate_provenance.py` with labelled development data to fit a
balanced logistic model using the four provenance features. It requires at
least 30 rows and both classes. Validate any artifact on held-out data before
calling it deployment-calibrated.

### Blind forensics

Blind forensics locates candidate manipulation regions from four complementary
signals. It does not determine whether an image is AI-generated by itself.

| Signal | Examines | Helps identify |
|---|---|---|
| Error-level analysis (ELA) | Differences after JPEG recompression | Different compression histories or inserted regions |
| High-pass/noise energy | Local residual/noise consistency | Uneven processing, smoothing, or synthesis clues |
| 8x8 JPEG grid | Block-grid consistency | JPEG processing discontinuities |
| ORB copy-move matching | Distant matching keypoints | Possible duplicated content |

The maps form a consensus candidate mask and up to 20 labelled regions (`R1`,
`R2`, …). Each region includes inclusive pixel and normalized boxes, location,
mean anomaly strength, and multi-signal support. The compact feature vector,
candidate coverage, and capped `integrity_risk_weight` support routing or later
calibration. The integrity weight is capped at 0.25 because normal textures,
camera pipelines, post-processing, and social-media compression can resemble
manipulation.

Tier 1 outputs three normal artifacts:

- `candidate_mask`: machine-readable localization aid;
- `vlm_overlay`: anomaly overlay intended as concise VLM context; and
- `human_review`: three-panel view with boxes, mask overlay, and anomaly heatmap.

`--debug-forensics` exposes raw maps, thresholds, signal values, timing, and
region crops for tuning only; these must not be passed to downstream models in
normal operation.

### Routing

The router uses a fast path only when `verified_ai_signal` is true: a verified
AI content credential or provider-confirmed AI watermark. It then skips Tier 2
and Tier 3 and returns `verified_provenance_fast_path`.

Trusted digital-capture provenance is retained as strong supporting evidence,
but still runs both later tiers. The same continuation policy applies to missing,
invalid, or unverified provenance and to forensic anomalies.

## Tier 2: Vision Language Model analysis

Tier 2 provides semantic analysis that a binary classifier and local pixel
heuristics cannot provide reliably. The VLM is asked to examine:

1. Lighting, shadows, and reflections.
2. Anatomy, text, patterns, and impossible geometry.
3. Background repetition, object consistency, and perspective.
4. Compression and noise, while distinguishing ordinary reposting artifacts
   from generative artifacts.

The structured response contains:

```json
{
  "source_estimate": "DALL-E | Midjourney | Stable Diffusion | Other AI | Real | Uncertain",
  "ai_coverage": 0.0,
  "confidence": 0.0,
  "reasoning": "plain-language explanation",
  "evidence": ["short observation"]
}
```

`ai_coverage` is the VLM's estimated fraction of the canvas that appears
AI-generated or AI-modified. It is not calculated from a forensic mask or
Grad-CAM. Returned labels are normalized to this allowed set and numbers are
clamped to `[0, 1]`; invalid or failed responses become uncertain rather than
becoming a false negative.

OpenAI and Gemini adapters run concurrently when their API keys are configured.
The aggregator selects the highest-confidence source/confidence, calculates
coverage as a confidence-weighted average, and retains evidence from every valid
provider. If all providers fail, Tier 2 returns `Uncertain`, no coverage, and
zero confidence.

The intended Tier 2 handoff includes the normalized image, forensic overlay,
candidate regions, and concise provenance state. Raw manifests, individual maps,
and the human review panel must not be supplied. The current OpenAI and Gemini
provider calls submit the image and prompt only; overlay/region/provenance
serialization remains an integration task and should not be presented as
currently implemented behavior.

## Tier 3: ConvNeXt detector and Grad-CAM

Tier 3 is an independent learned visual detector. It consumes only the
normalized RGB image, keeping it independent from Tier 1 provenance and
handcrafted forensic features.

```text
Normalized RGB image
    -> Albumentations robustness transforms
    -> 224x224 ImageNet-normalized tensor
    -> pretrained ConvNeXt-Tiny backbone
    -> pooling / classifier path -> dropout -> one logit
    -> sigmoid P(AIGC)
```

ConvNeXt-Tiny is well below the challenge's two-billion-parameter limit. The
classifier head is trainable and the default setup selectively unfreezes the
last two ConvNeXt stages. One stage is safer/faster, two is the recommended
starting point, and three offers more adaptation with more overfitting risk.
Full-network fine-tuning is not the starting configuration.

### Training and robustness

The classifier uses label `0` for real/authentic and `1` for AI-generated.
Training manifests provide `image_path`, `label`, `source`, and `split`.
WildFake remains validation-only and must not influence training, threshold
tuning, or repeated model selection.

Default differential learning rates protect pretrained features while adapting
the classifier: `1e-5` for the backbone and `2e-4` for the head. Stochastic
augmentations emulate the transformations expected after social-media sharing:

- JPEG quality 30–90;
- Gaussian blur sigma 0.5–2.0;
- downscale to 0.25x–0.5x then upscale;
- Gaussian noise 0.02–0.10;
- roughly ±20% colour jitter; and
- 80% centre crop.

Optional two-view consistency regularization trains on clean and transformed
views with `BCE(clean) + BCE(transformed) + lambda * MSE(P(clean),
P(transformed))`. The clean probability is detached, making the transformed
branch follow it. Begin with `consistency_weight: 0.0`, then evaluate `0.02`,
`0.05`, and `0.1` only if transformed-set performance justifies another forward
pass.

Record validation loss, ROC-AUC, average precision, F1, balanced accuracy,
accuracy, runtime, learning rates, and consistency loss. Balanced accuracy is
the default checkpoint-selection metric because raw accuracy can hide class
imbalance. Use a staged tuning funnel: short trials for unfreezing depth,
focused learning-rate trials, robustness trials, then one full final run.

### Grad-CAM

Grad-CAM uses gradients of the AIGC logit at the final ConvNeXt feature block's
depthwise convolution to create a heatmap. Bright regions contributed more to
the AI logit; dark regions contributed less. It is an explanation of the
classifier, not a pixel-level segmentation model. Do not use it as `ai_coverage`
or claim it measures an exact generated percentage.

## Fusion and UI

Fusion consumes compact signals:

```text
provenance severity + verified-AI flag + forensic integrity weight
    + VLM confidence + ConvNeXt probability
    -> held-out calibrated fusion policy
```

Do not directly add scores, especially correlated provenance checks. A fitted
fusion/calibration policy must be trained and evaluated on held-out data, and
its method, calibration status, threshold, inputs, and latency are stored in the
`fusion` block.

Before such a model is supplied, the orchestrator uses an interim uncalibrated
policy: use the available VLM or ConvNeXt probability alone, or calculate
`0.65 × ConvNeXt + 0.35 × VLM` when both are available. A VLM result contributes
`confidence` for an AI source and `1 - confidence` for `Real`; `Uncertain`
contributes no VLM probability. The provisional decision threshold is 0.5.

The UI should display final verdict and confidence, source and coverage where
available, concise provenance conclusion, C2PA history, evidence, Tier 1 human
review artifact, and Tier 3 heatmap. Keep raw manifests, validation diagnostics,
metadata fields, thresholds, and individual maps behind developer mode.

## Setup and commands

From the repository root:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Copy `.env.example` to `.env` and configure keys as required:

- `OPENAI_API_KEY`: OpenAI provenance and OpenAI VLM;
- `GEMINI_API_KEY`: Gemini VLM; and
- `ANTHROPIC_API_KEY`: reserved by the current setup configuration.

Run the UI:

```powershell
python -m streamlit run .\apps\demo\app.py
```

Run Tier 1 on untouched bytes:

```powershell
$env:PYTHONPATH = "src"
python -m truesight.provenance.tier1 data\image.jpg
```

Useful Tier 1 flags: `--skip-openai-provenance`, `--skip-forensics`,
`--debug-provenance`, `--debug-forensics`, and `--calibration PATH`.

Run full directory evaluation:

```powershell
python scripts\evaluate.py --input_dir data\samples --output predictions.json
```

Training, prediction, and Grad-CAM commands are available in
`scripts/ConvNext/`. Dataset download and manifest preparation utilities are in
`scripts/data/`.

## Reproducing the model results

### 1. Prepare data

Download the supported datasets (CIFAKE and SID-Set) if they are not already
available locally:

```powershell
python scripts\data\download_datasets.py
```

Prepare independent CSV manifests with the following minimum columns:

```csv
image_path,label,source,split
/absolute/path/real_001.jpg,0,real_dataset,train
/absolute/path/ai_001.jpg,1,cifake,train
```

Keep the final benchmark set, including WildFake, separate from the training
and tuning manifests. Create a clean validation manifest and a corresponding
transformed manifest containing the same labels and class distribution. The
transformed split should apply JPEG, blur, resize, noise, crop, and colour
adjustment independently and in representative combinations.

### 2. Train ConvNeXt

Run training from the repository root. The following command uses the default
ConvNeXt-Tiny configuration and writes a reproducible checkpoint and training
history to the selected output directory:

```powershell
python scripts\ConvNext\train.py `
  --manifest data\processed\train_manifest.csv `
  --val-manifest data\processed\val_clean_manifest.csv `
  --config configs\model\convnext_tiny.yaml `
  --output-dir outputs\convnext_tiny
```

The selected checkpoint is `outputs\convnext_tiny\best.pt`. Record the config,
manifest versions, random seed, checkpoint, threshold, and commit used for each
experiment. This prevents accidental comparison of results produced with
different data splits or preprocessing.

### 3. Evaluate clean and transformed splits

Evaluate the exact same checkpoint and decision threshold on each manifest:

```powershell
python scripts\evaluate_manifest.py `
  --checkpoint outputs\convnext_tiny\best.pt `
  --manifest data\processed\val_clean_manifest.csv `
  --threshold 0.5 `
  --output-json outputs\metrics\clean.json

python scripts\evaluate_manifest.py `
  --checkpoint outputs\convnext_tiny\best.pt `
  --manifest data\processed\val_transformed_manifest.csv `
  --threshold 0.5 `
  --output-json outputs\metrics\transformed.json
```

`evaluate_manifest.py` saves the confusion matrix, per-class precision/recall/F1,
and aggregate metrics. It also makes the false-positive and false-negative
counts directly available for review:

```text
false_positive = true_real predicted as AI
false_negative = true_AI predicted as real
```

For end-to-end pipeline evaluation, use `scripts\evaluate.py` after setting
the VLM/provenance keys and checkpoint environment configuration. Its output is
the compact prediction JSON defined by `schemas/evaluation.schema.json`.

## Robustness evaluation summary

The checked-in two-stage ConvNeXt-Tiny experiment (`unfreeze_stages: 2`, five
epochs, threshold 0.5) achieved the following **clean validation** result at
its selected epoch. The values are taken from the experiment's checked-in
training-history artifact.

| Evaluation split | Accuracy | Balanced accuracy | F1 | ROC-AUC | Average precision | Status |
|---|---:|---:|---:|---:|---:|---|
| Clean validation | 95.48% | 95.48% | 95.45% | 99.20% | 99.24% | Measured |
| Transformed validation | — | — | — | — | — | Not yet recorded in a checked-in evaluation artifact |

The transformed row is intentionally not estimated from clean validation. A
robustness claim needs the same checkpoint evaluated on a held-out transformed
manifest using the commands above. When that run is complete, replace the
second row with the generated values and retain `clean.json` and
`transformed.json` with the submission artifacts.

For the final report, additionally break transformed performance down by
corruption family. This compact matrix makes regressions easy to spot:

| Split / transformation | Accuracy | Balanced accuracy | F1 | Notes |
|---|---:|---:|---:|---|
| Clean | from `clean.json` | from `clean.json` | from `clean.json` | Baseline |
| JPEG compression | from transformed evaluation | from transformed evaluation | from transformed evaluation | Quality 30–90 |
| Gaussian blur | from transformed evaluation | from transformed evaluation | from transformed evaluation | Sigma 0.5–2.0 |
| Downscale + upscale | from transformed evaluation | from transformed evaluation | from transformed evaluation | 0.25x–0.5x |
| Gaussian noise | from transformed evaluation | from transformed evaluation | from transformed evaluation | Std. dev. 0.02–0.10 |
| Colour adjustment | from transformed evaluation | from transformed evaluation | from transformed evaluation | Approximately ±20% |
| Centre crop | from transformed evaluation | from transformed evaluation | from transformed evaluation | 80% crop |

## Error analysis

No ranked false-positive/false-negative artifact is currently checked into the
repository, so the cases below are representative risk categories rather than
claims about a measured error count. After each evaluation run, inspect images
identified by `false_positive` and `false_negative` in the saved confusion
matrix, then save the reviewed examples and their heatmaps with the experiment.

| Error type | Representative case | Why it can fail | Mitigation / trade-off |
|---|---|---|
| False positive | A real low-light, highly compressed, or heavily filtered social-media photo | Noise, JPEG blocks, sharpening, and unusual texture can resemble generative artifacts | Stronger real-camera and social-media data reduces these errors, but overly aggressive augmentation can dilute useful AI cues |
| False positive | A legitimate composite, studio image, or repeated natural pattern | Blind-forensic maps can flag edit boundaries or repeated texture without proving AI origin | Keep forensic integrity capped and require Tier 2/3 confirmation; this preserves precision but may add latency |
| False negative | A high-quality modern AI image with no provenance and realistic anatomy/lighting | Generator artifacts may be subtle or outside the training distribution | Broaden generator coverage and use a held-out, recent benchmark; this increases data and training cost |
| False negative | An AI image after strong blur, resize, or recompression | Transformations erase high-frequency visual evidence used by the classifier | Robust augmentation and consistency training improve resilience, but can lower clean-image specialization |
| Inconclusive provenance | An image whose credentials were stripped in transit | Absence of C2PA/watermark is not evidence of authenticity | Continue to VLM and ConvNeXt; this is safer but slower than a provenance fast path |

The key trade-off is deliberate: TrueSight prioritizes calibrated evidence and
uncertainty over a fast but overconfident decision. Provenance can be decisive
when verified; in all other cases, the pipeline spends more compute to reduce
the risk of a single weak signal deciding the result.

## Limitations and improvements with more time

Current limitations include stripped credentials, compression-driven forensic
false positives, VLM hallucinations and latency, and ConvNeXt dataset bias or
adversarial vulnerability. The present fusion weights are an interim,
uncalibrated policy; they are not a substitute for held-out calibration.

With more time, the highest-priority improvements would be:

1. Run and publish the full clean-versus-transformed benchmark, including
   per-corruption metrics, confidence intervals, and reviewed error cases.
2. Fit the provenance and final fusion policies on held-out data, then monitor
   calibration drift as new generators appear.
3. Expand training data across current generators, real camera pipelines,
   legitimate edited photographs, and platform recompression chains.
4. Add an automated error-review report that exports the highest-confidence
   false positives and false negatives alongside Grad-CAM and forensic artifacts.
5. Evaluate adversarial robustness, throughput, provider failure handling, and
   privacy/cost controls before production deployment.

## Team contributions

| Name | Area | Contribution | 
|---|---|---|
|Zowie| Computer vision and model | ConvNeXt transfer learning, Albumentations robustness pipeline, selective fine-tuning, inference, and Grad-CAM heatmaps |
|Jing En | Provenance and watermarking | C2PA/JUMBF parsing, provenance-chain validation, metadata checks, provider provenance integration, severity signals, and Tier 1 routing |
|Yong Lin| VLM analysis | Asynchronous OpenAI/Gemini visual analysis, structured JSON parsing, source attribution, coverage estimation, and evidence prompts |
|Sania| Integration and application | Pipeline orchestration, result schema, adapters, batch evaluation interface, Streamlit demo, repository structure, and README reproduction workflow |
|Aadith| Benchmarking and demo | Dataset preparation, clean/transformed evaluation design, robustness reporting, error-review workflow, demo-video preparation, and submission materials |

## Project references

The detailed supporting materials remain available in `docs/`:

- `docs/architecture.md`
- `docs/team_integration.md`
- `docs/api_contract.md`
- `docs/tier1_provenance_guide.md`
- `docs/blind_forensics.md`
- `docs/tuning_checklist.md`
- `docs/setup.md`
- `docs/experiments.md`
- `docs/limitations.md`
- `docs/video.md`

Machine-readable contracts are in `schemas/prediction.schema.json` and
`schemas/provenance.schema.json`.
