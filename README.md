# TrueSight

TrueSight is a three-tier prototype for detecting AI-generated and AI-modified images under realistic image transformations.

It combines:

1. cryptographically verifiable provenance;
2. semantic visual reasoning; and
3. an independent learned visual classifier.

The tiers deliberately use different evidence sources so that a weakness in one method does not become the single point of failure for the entire system.

---

## Architecture

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
           normalized derivative; original file remains untouched
                        |                         |
                        v                         v
           Tier 2: VLM semantic analysis   Tier 3: ConvNeXt-Tiny
           source / coverage / evidence    P(AIGC) + Grad-CAM
                        \                         /
                         \                       /
                          v                     v
                        calibrated fusion and UI result

The original uploaded file must never be re-encoded before Tier 1. Resizing, RGB conversion, EXIF handling, and recompression can remove or invalidate provenance evidence.
Tier 2 and Tier 3 use a separate normalized derivative. The original file is retained for provenance checks and audit.

### Result contract and ownership

`schemas/prediction.schema.json` is the canonical result contract.
Tier 1 creates the complete envelope with tier2, tier3, and fusion set to null.
Each tier writes only its own block. Fusion writes the final top-level verdict.

| Component | Produces | Does not do |
| --- | --- | --- |
| Tier 1 | Provenance, forensic artifacts, routing signals | Re-encode the source or infer authenticity from missing metadata |
| Tier 2 | VLM verdict, source estimate, AI coverage, semantic evidence | Treat absent provenance as proof of a real image |
| Tier 3 | ConvNeXt visual probability and Grad-CAM heatmap | Use provenance or VLM results as model inputs |
| Fusion | Final verdict, confidence, threshold, calibration record | Add correlated scores directly |
| UI | Human-readable evidence and review artifacts | Show raw debug data by default |

Use `null` for unknown, unavailable, or not-yet-produced values.
Use `false` only after a check actually ran and returned a negative result.
Missing C2PA, for example, is not evidence that an image is authentic.

## Tier 1: provenance and blind forensics

Tier 1 has two separate roles:
- verify attributable provenance evidence in the untouched file;
- produce weak local anomaly evidence without requiring an original/reference image.

Only verified AI-attributed provenance can skip the later tiers.
Verified camera capture, missing provenance, invalid credentials, unverified marks, and forensic anomalies remain inconclusive and continue downstream.

### C2PA content credentials
C2PA content credentials are signed provenance records embedded in an asset.
TrueSight reads the manifest store and returns:
- the active/root manifest and reachable ingredient manifests;
- ingredient relationships, missing references, cycles, and unsigned ingredients;
- declared actions from oldest to active asset;
- origin/digital-source type;
- declared edits and transformations;
- metadata edits; and
- validation state and signer trust.

The checker separates credentials that merely parse from credentials trusted by policy.
Only the C2PA SDK's trusted validation state is considered verified. A valid but untrusted signer is not sufficient.
A C2PA history tells us what the signed history declares. It does not prove that no undeclared edit occurred after signing.

### Provider provenance, watermarks, and metadata
With `OPENAI_API_KEY` configured, Tier 1 can send untouched bytes to the configured OpenAI provenance check.
It may report supported C2PA and image watermark signals.
Every source is normalized to a common signal with:
- kind;
- provider;
- status;
- present;
- verified;
- ai_claim;
- optional source;
- verification basis; and
- evidence score.

Important states are:
- `unavailable`: no API key or dependency is available;
- `not_checked`: the check was explicitly skipped;
- `not_detected`: the completed check found no supported signal;
- `verified`: a verified AI-attributed credential or watermark was found.

Embedded metadata such as a Software tag is weak evidence because it can be forged, edited, or stripped.

### Severity and calibration
The default provenance policy avoids double-counting correlated local and provider checks of the same credential.
These are evidence contributions, not calibrated real-world probabilities:

| Signal | Default contribution |
| --- | --- |
| Verified AI content credential | 0.98 |
| Provider-confirmed AI watermark | 0.97 |
| Present but unverified AI credential | 0.35 |
| AI-related embedded metadata | 0.15 |

Use `scripts/calibrate_provenance.py` with labelled development data to fit a balanced logistic model using the provenance features.
Validate calibration artifacts on held-out data before calling them deployment-calibrated.

### Blind forensics
Blind forensics locates candidate manipulation regions from several signals:

| Signal | Examines | Helps identify |
| --- | --- | --- |
| Error-level analysis | Differences after JPEG recompression | Different compression histories |
| High-pass/noise energy | Local residual consistency | Uneven processing or synthesis clues |
| JPEG grid analysis | Block-grid consistency | JPEG processing discontinuities |
| ORB matching | Repeated keypoints | Possible duplicated content |

These maps form a consensus candidate mask and labelled regions.
Blind forensics provides evidence for review and routing. It does not determine whether an image is AI-generated by itself.

### Routing
The router uses a fast path only when `verified_ai_signal` is true.
It then returns a verified-provenance result and can skip later tiers.
Missing, invalid, or unverified provenance continues to Tier 2 and Tier 3.

## Tier 2: Vision Language Model analysis

Tier 2 provides semantic analysis that a binary classifier and local pixel heuristics cannot reliably provide on their own.
The VLM is asked to examine:
- lighting, shadows, and reflections;
- anatomy, text, patterns, and impossible geometry;
- background repetition, object consistency, and perspective;
- compression and noise while distinguishing reposting artifacts from generative artifacts.

A structured response contains:
    
    {
      "source_estimate": "DALL-E | Midjourney | Stable Diffusion | Other AI | Real | Uncertain",
      "ai_coverage": 0.0,
      "confidence": 0.0,
      "reasoning": "plain-language explanation",
      "evidence": ["short observation"]
    }

`ai_coverage` is the VLM's estimated fraction of the canvas that appears AI-generated or AI-modified. It is not calculated from a forensic mask or Grad-CAM.
If provider calls fail or return invalid data, the result becomes uncertain rather than automatically becoming a false negative.

## Tier 3: ConvNeXt visual detector

Tier 3 is an independent learned visual detector.
It consumes only a normalized RGB image:

    Normalized RGB image
        ↓
    Resize/crop
        ↓
    Optional robustness transformations
        ↓
    ImageNet normalization
        ↓
    Pretrained ConvNeXt-Tiny backbone
        ↓
    Global feature representation
        ↓
    Dropout
        ↓
    One-logit binary classifier
        ↓
    Sigmoid P(AIGC)

The binary labels are:
- 0 = real/authentic
- 1 = AI-generated or AI-manipulated

The output is a visual model score:
`pred = P(AIGC | image)`

It should not be described as proof of authorship or as a universally calibrated probability.

### Why ConvNeXt-Tiny?
ConvNeXt-Tiny is a modern convolutional architecture suitable for the challenge's image-level classification task.
It was selected because:
- it uses publicly available ImageNet pretrained weights;
- it provides a strong visual representation without training from scratch;
- it is practical for hackathon-scale compute;
- it supports selective fine-tuning;
- its feature maps support Grad-CAM;
- it remains far below the challenge's two-billion-parameter limit.

The original TorchVision ConvNeXt-Tiny has approximately 28.6 million parameters. This project replaces its original 1,000-class ImageNet classifier with a one-logit binary classifier, resulting in approximately 27.8 million parameters.

### Transfer learning and selective fine-tuning
Training starts with the pretrained ConvNeXt-Tiny representation.
The classifier head is replaced with:
    
    768-dimensional ConvNeXt representation
        ↓
    Dropout
        ↓
    Linear layer
        ↓
    One output logit

Training uses two phases.

**Classifier warm-up**
Initially:
- ConvNeXt stages: frozen
- Binary classifier: trainable

This adapts the new classifier without immediately changing the pretrained feature extractor.

**Selective fine-tuning**
After warm-up, the final selected ConvNeXt stages are unfrozen.
Configuration:
    
    model:
      unfreeze_stages: 2

Interpretation:
- 0 = classifier only
- 1 = last stage
- 2 = last two stages
- 3 = last three stages
- 4 = all four stages

The classifier uses a larger learning rate than the pretrained backbone:
    
    training:
      backbone_lr: 0.00001
      head_lr: 0.0002

During warm-up, a log may show approximately:
`trainable = 2,305`
This is expected because only the binary classifier is active. After warm-up, the selected ConvNeXt stages become trainable.

### Robustness training
The challenge specifically considers robustness against:
- JPEG compression;
- Gaussian blur;
- image resizing;
- Gaussian noise;
- color adjustment;
- center cropping.

The augmentation implementation is in:
`src/truesight/vision/augmentations.py`

The default ranges are:

| Transformation | Range | Real-world motivation |
| --- | --- | --- |
| JPEG compression | quality 30–90 | Social-media re-encoding |
| Gaussian blur | sigma 0.5–2.0 | Processing or focus blur |
| Downscale/upscale | 0.25–0.5 scale | Thumbnail generation |
| Gaussian noise | 0.02–0.10 | Sensor or transmission noise |
| Color jitter | approximately ±20% | Filters and auto-enhancement |
| Center crop | approximately 80% | Reframing and profile-picture cropping |

The pipeline does not apply every severe corruption to every image. A corruption family is sampled so the training distribution remains realistic.
Enable robustness training with:
    
    augmentation:
      enabled: true

The validation pipeline remains deterministic and does not use random corruption.

### Consistency regularization
The optional consistency objective presents two views of the same image:
    
    same source image
        ├── normal view
        └── transformed view

The model is trained with:
`BCE(normal view, label) + BCE(transformed view, label) + lambda × consistency loss`

The consistency term encourages similar image-level predictions after realistic transformations.
For example:
- Original image: P(AI) = 0.92
- Compressed image: P(AI) = 0.88
is reasonably stable.

By contrast:
- Original image: P(AI) = 0.92
- Compressed image: P(AI) = 0.31
suggests dependence on fragile evidence.

Consistency regularization does not estimate the percentage of pixels generated by AI. It is an image-level training objective.

Initial configuration:
    
    training:
      consistency_weight: 0.0
      use_consistency_view: false

Experimental configuration:
    
    training:
      consistency_weight: 0.05
      use_consistency_view: true

Because it requires another forward pass, it increases training cost.

### Grad-CAM explainability
Tier 3 includes Grad-CAM to provide a visual explanation of the classifier.
The heatmap highlights regions that contributed to the AI-generation logit.
The correct interpretation is:
*Regions influencing the model's AI-generation prediction.*

Grad-CAM is not:
- pixel-level ground-truth segmentation;
- an exact map of generated pixels;
- an estimate of the percentage of the image generated by AI;
- independent evidence that a highlighted region is synthetic.

The UI should display the heatmap as supporting model evidence rather than definitive proof.

Example:
    
    python -u scripts\convnext\gradcam.py `
        --checkpoint outputs\member1\convnext_tiny\best.pt `
        --image data\samples\example.jpg `
        --output outputs\member1\heatmaps\example.png


### Tier 3 output contract
Tier 3 returns visual evidence in a compact form:
    
    {
      "image_path": "data/test/example.jpg",
      "pred": 0.892,
      "model": "convnext_tiny",
      "model_version": "member1-exp02"
    }

With Grad-CAM:
    
    {
      "image_path": "data/test/example.jpg",
      "pred": 0.892,
      "model": "convnext_tiny",
      "model_version": "member1-exp02",
      "heatmap_path": "outputs/member1/heatmaps/example.png"
    }


Tier 3 does not read provenance, VLM, or forensic fields. This independence allows Fusion to combine different evidence sources later.

### Dataset contract
Member 1 consumes CSV manifests.
Required columns:
    
    image_path,label,source,split
    /path/to/img001.jpg,0,CIFAKE,train
    /path/to/img002.jpg,1,CIFAKE,train

Labels:
- 0 = real/authentic
- 1 = AI-generated or AI-manipulated

The model does not hard-code CIFAKE, SID-Set, or any other specific dataset. This allows Member 5 to change the data source without changing the ConvNeXt implementation.
WildFake remains validation-only and must not be included in training manifests, augmentation tuning, threshold tuning, or repeated model selection.

### Training strategy
Use a staged experiment funnel.

- **Experiment 1: clean baseline**
  - ConvNeXt-Tiny
  - 224 × 224
  - pretrained weights
  - last 1 stage unfrozen
  - augmentation disabled
  - consistency disabled
- **Experiment 2: deeper fine-tuning**
  - last 2 stages unfrozen
  - augmentation disabled
  - consistency disabled
- **Experiment 3: robustness**
  - last 2 stages unfrozen
  - augmentation enabled
  - consistency disabled
- **Experiment 4: consistency**
  - last 2 stages unfrozen
  - augmentation enabled
  - consistency enabled
  - consistency weight 0.02 or 0.05

Record:
- accuracy;
- balanced accuracy;
- F1-score;
- ROC-AUC;
- average precision;
- false-positive count;
- false-negative count;
- validation loss;
- training time;
- learning rates;
- consistency loss.

Do not select the final model using a small manually selected image folder.
Use a complete validation manifest and perform the external WildFake check only after the training recipe has been selected.

### Evaluation discipline
Raw accuracy can hide class imbalance, so balanced accuracy is used as the default checkpoint-selection metric. For a balanced validation set, accuracy and balanced accuracy may be identical.

A small sample can produce a misleading impression. For example, three false positives in a manually selected ten-image folder does not prove that the model has a 30% false-positive rate.

Use:
    
    full validation manifest
        ↓
    complete confusion matrix
        ↓
    precision, recall, F1, ROC-AUC
        ↓
    false-positive and false-negative inspection

Report both clean and transformed performance whenever possible.

## Fusion and UI

Fusion consumes compact signals:
    
    provenance severity
    + verified-AI flag
    + forensic integrity weight
    + VLM confidence
    + ConvNeXt probability
        ↓
    calibrated fusion policy
        ↓
    final verdict


Do not directly add correlated scores.
Before a calibrated fusion model is available, an interim policy can use:
*ConvNeXt probability alone*

or, when both signals are available:
*0.65 × ConvNeXt probability + 0.35 × VLM probability*

This interim policy is not a deployment-calibrated probability.

The UI should distinguish:
*ConvNeXt visual score*
from:
*final fused TrueSight verdict*

The UI may display:
- final verdict;
- confidence;
- source estimate;
- VLM coverage;
- provenance conclusion;
- C2PA history;
- forensic review artifact;
- ConvNeXt probability;
- Grad-CAM heatmap.

Raw manifests, thresholds, debug maps, and validation diagnostics should remain behind developer mode.

## Project structure

    truesight/
    │
    ├── apps/
    │   └── demo/
    │
    ├── configs/
    │   └── model/
    │       └── convnext_tiny.yaml
    │
    ├── scripts/
    │   ├── convnext/
    │   │   ├── train.py
    │   │   ├── predict.py
    │   │   ├── gradcam.py
    │   │   └── tune.py
    │   │
    │   └── data/
    │       ├── download_datasets.py
    │       └── make_experiment_manifests.py
    │
    ├── src/
    │   └── truesight/
    │       ├── vision/
    │       ├── provenance/
    │       ├── vlm/
    │       └── fusion/
    │
    ├── data/
    │   ├── raw/
    │   └── processed/
    │
    ├── outputs/
    ├── models/
    ├── schemas/
    ├── tests/
    ├── docs/
    ├── requirements.txt
    └── README.md

Dataset files, generated outputs, and large model checkpoints should not normally be committed directly to Git. Use Git LFS or a separate artifact store for large checkpoints when necessary.

## Installation

From the repository root:
    
    python -m venv .venv
    .venv\Scripts\Activate.ps1
    python -m pip install --upgrade pip
    python -m pip install -r requirements.txt


Set environment variables in `.env` as required:
    
    OPENAI_API_KEY=
    GEMINI_API_KEY=
    ANTHROPIC_API_KEY=


Run the UI:
    
    python -m streamlit run .\apps\demo\app.py


### ConvNeXt commands
Set the import path:
    
    $env:PYTHONPATH = "$((Get-Location).Path)\src;$env:PYTHONPATH"


Train:
    
    python -u scripts\convnext\train.py `
        --config configs\model\convnext_tiny.yaml `
        --manifest data\processed\train_manifest.csv `
        --val-manifest data\processed\val_manifest.csv `
        --output-dir outputs\member1\convnext_tiny


Run directory inference:
    
    python -u scripts\convnext\predict.py `
        --checkpoint outputs\member1\convnext_tiny\best.pt `
        --input-dir data\samples `
        --output-json outputs\member1\predictions.json


Generate Grad-CAM:
    
    python -u scripts\convnext\gradcam.py `
        --checkpoint outputs\member1\convnext_tiny\best.pt `
        --image data\samples\example.jpg `
        --output outputs\member1\heatmaps\example.png


## Limitations

TrueSight is a hackathon-scale prototype.
Known limitations include:
- dataset-specific shortcuts;
- possible cross-dataset generalisation failure;
- false positives on unusual authentic images;
- false negatives on unfamiliar AI generators;
- threshold sensitivity;
- incomplete probability calibration;
- limited CPU training speed;
- Grad-CAM's limited spatial precision;
- possible correlation between VLM, provenance, and visual evidence;
- incomplete validation under all real-world platform processing pipelines.

The system should be presented as decision support and evidence aggregation, not as definitive proof of image authorship.

encourages the model to maintain its image-level prediction when the same image is redistributed through these transformations. Grad-CAM then provides a visual explanation of the regions that influenced the model's prediction. The ConvNeXt score remains separate from provenance and VLM evidence so that the final system can combine complementary signals through a later fusion layer.*
