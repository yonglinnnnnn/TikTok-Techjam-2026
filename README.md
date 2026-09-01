# TrueSight

TrueSight is a three-tier prototype for detecting AI-generated and AI-modified images under realistic image transformations.

## 1. Project overview

TrueSight combines complementary evidence sources:

1. **Tier 1 - Provenance and blind forensics**
   - C2PA content credentials;
   - provider-confirmed provenance and watermark signals;
   - metadata inspection; and
   - blind forensic localization using ELA, high-pass/noise analysis, JPEG-grid analysis, and ORB copy-move matching.

2. **Tier 2 - Vision Language Model analysis**
   - semantic inspection of lighting, anatomy, text, reflections, perspective, repetition, and impossible geometry;
   - source estimate;
   - confidence;
   - evidence explanation; and
   - estimated AI coverage.

3. **Tier 3 - ConvNeXt visual detection**
   - pretrained ConvNeXt-Tiny;
   - real-versus-AI image classification;
   - robustness augmentation;
   - optional consistency training; and
   - Grad-CAM model explanation.

The tiers are deliberately separated. Tier 3 does not consume provenance or VLM output unless a separately trained multi-input model is created. Fusion combines compact signals after each tier has completed.

## 2. Problem addressed

AI-generated images are becoming increasingly realistic. A detector that performs well only on pristine laboratory images may fail after an image is compressed, blurred, resized, cropped, recolored, or reposted.

TrueSight addresses this by:

- preserving the original bytes for provenance checks;
- creating a separate normalized derivative for visual models;
- training ConvNeXt with realistic redistribution transformations;
- using VLM reasoning for semantic inconsistencies;
- using blind forensics for candidate manipulation regions;
- retaining false-positive and false-negative evidence for review; and
- combining independent signals through a final fusion layer.

The system is a hackathon-scale proof of concept, not a definitive authorship oracle.

## 3. Development tools, models, APIs, and libraries

### Development tools

- Visual Studio Code;
- Windows PowerShell;
- Python virtual environments;
- Jupyter/Colab-compatible training workflow;
- Git and GitHub;
- local CPU or CUDA-enabled PyTorch execution.

### Models and APIs

- ConvNeXt-Tiny with TorchVision ImageNet weights;
- OpenAI content provenance check, when configured;
- OpenAI Vision Language Model adapter;
- Gemini Vision Language Model adapter;
- optional Anthropic integration point;
- C2PA SDK/content-credentials tooling.

An unavailable API produces an unavailable result. It must not be treated as evidence that an image is authentic.

### Main libraries

- PyTorch;
- TorchVision;
- Albumentations;
- OpenCV;
- Pillow;
- NumPy;
- pandas;
- scikit-learn;
- PyYAML;
- Hugging Face datasets/huggingface_hub;
- KaggleHub;
- tqdm;
- Streamlit.

## 4. Datasets and assets

The project supports:

- CIFAKE: real and AI-generated images downloaded from Kaggle;
- SID-Set: real, full-synthetic, and tampered images from Hugging Face;
- WildFake subset: external validation/reference data only.

SID-Set labels are mapped as follows:

~~~~text
0 = real
1 = full synthetic -> binary AI label 1
2 = tampered      -> binary AI label 1
~~~~

WildFake must not be used for training, augmentation tuning, threshold tuning, or repeated model selection.

Dataset files are intentionally excluded from normal Git commits. They are downloaded locally using scripts/data/download_datasets.py and represented by CSV manifests in data/processed/.

## 5. Architecture and data lifecycle

~~~~text
Original uploaded bytes
        |
        +--> Tier 1: provenance + metadata + blind forensics
        |
        +--> normalized derivative
                |
                +--> Tier 2: VLM semantic analysis
                |
                +--> Tier 3: ConvNeXt-Tiny + Grad-CAM
                                |
                                v
                         Fusion and UI
~~~~

The original uploaded file is never overwritten. RGB conversion, resizing, EXIF handling, and recompression happen only on the derivative used by later tiers.

## 6. Tier 1: provenance and blind forensics

Tier 1 checks signed provenance and produces weak forensic evidence.

C2PA output includes the active/root manifest, reachable ingredients, relationships, declared actions, transformations, metadata edits, validation state, and signer trust.

Stable provenance statuses include:

- verified;
- detected;
- not_detected;
- not_present;
- invalid;
- unavailable;
- error; and
- not_checked.

Only a verified AI-attributed content credential or provider-confirmed watermark can activate the fast path. Missing provenance does not establish authenticity.

Blind forensics uses:

- JPEG recompression/error-level differences;
- local high-pass/noise inconsistency;
- 8x8 JPEG-grid inconsistency; and
- spatially separated ORB matches.

These outputs are candidate review evidence. They are not independently conclusive and are not sent directly to ConvNeXt.

## 7. Tier 2: VLM analysis

The VLM examines:

- lighting, shadows, and reflections;
- anatomy, text, patterns, and impossible geometry;
- repeated background content;
- object consistency and perspective;
- compression and noise; and
- likely image source.

Example output:

~~~~json
{
  "source_estimate": "Other AI",
  "ai_coverage": 0.75,
  "confidence": 0.82,
  "reasoning": "Short explanation",
  "evidence": ["Short visual observation"]
}
~~~~

ai_coverage is a semantic estimate from the VLM. It is not derived from Grad-CAM or a forensic mask.

## 8. Tier 3: ConvNeXt detector

ConvNeXt-Tiny receives a normalized RGB image and returns an image-level AIGC score.

~~~~text
RGB image
  -> resize/crop
  -> optional robustness transformation
  -> ImageNet normalization
  -> pretrained ConvNeXt-Tiny
  -> dropout and one-logit classifier
  -> sigmoid P(AIGC)
~~~~

Labels are:

~~~~text
0 = real/authentic
1 = AI-generated or AI-manipulated
~~~~

The original TorchVision ConvNeXt-Tiny has approximately 28.6 million parameters. Replacing the 1,000-class ImageNet head with a one-logit classifier produces an adapted model of approximately 27.8 million parameters, below the two-billion-parameter constraint.

### Transfer learning

Training begins with ImageNet weights.

During classifier warm-up, only the binary head is trainable. A log showing approximately 2,305 trainable parameters is therefore expected.

After warm-up, the final selected ConvNeXt stages are unfrozen:

~~~~yaml
model:
  unfreeze_stages: 2
~~~~

The selected backbone uses a smaller learning rate than the classifier:

~~~~yaml
training:
  backbone_lr: 0.00001
  head_lr: 0.0002
~~~~

### Robustness augmentation

The training pipeline simulates:

| Transformation | Range | Real-world analogy |
|---|---:|---|
| JPEG compression | quality 30-90 | Social-media re-encoding |
| Gaussian blur | sigma 0.5-2.0 | Processing or focus blur |
| Downscale/upscale | 0.25-0.5 scale | Thumbnail repost |
| Gaussian noise | 0.02-0.10 | Transmission noise |
| Color jitter | approximately plus/minus 20% | Filters |
| Center crop | approximately 80% | Reframing |

A corruption family is sampled instead of applying every severe transformation simultaneously. Validation remains deterministic.

### Consistency training

The optional consistency objective trains on normal and transformed views of the same image:

~~~~text
BCE(normal view, label)
+ BCE(transformed view, label)
+ lambda * consistency loss
~~~~

It encourages stable image-level predictions after realistic transformations. It is not segmentation and does not estimate generated-pixel coverage.

### Grad-CAM

Grad-CAM shows regions that contributed to the classifier's AI-generation prediction.

It is not an exact generated-pixel map, a percentage of generated content, or proof that every highlighted region is synthetic. The UI should label it: Regions influencing the AI prediction.

## 9. API contract and team integration

Tier 3 returns:

~~~~json
{
  "image_path": "data/test/example.jpg",
  "pred": 0.892,
  "model": "convnext_tiny",
  "model_version": "member1-exp02",
  "heatmap_path": null
}
~~~~

The canonical envelope is schemas/prediction.schema.json.

Tier 1 initializes the envelope. Each tier writes only its own block. Fusion writes the final top-level verdict.

Unknown is null, never a false negative.

The fusion layer uses compact independent features such as:

~~~~python
{
    "provenance_severity": result["tier1"]["severity_weight"],
    "provenance_verified_ai": result["tier1"]["verified_ai_signal"],
    "forensic_integrity": result["tier1"]["forensic_integrity_weight"],
    "vlm_confidence": result["tier2"]["confidence"],
    "convnext_probability": result["tier3"]["probability"],
}
~~~~

The three numeric scores must not simply be added. A final fusion/calibration policy should be fitted and evaluated on held-out data.

## 10. Repository structure

~~~~text
truesight/
├── apps/demo/
├── configs/model/convnext_tiny.yaml
├── scripts/
│   ├── convnext/train.py
│   ├── convnext/predict.py
│   ├── convnext/gradcam.py
│   ├── convnext/tune.py
│   ├── data/download_datasets.py
│   └── data/make_experiment_manifests.py
├── src/truesight/
│   ├── vision/
│   ├── provenance/
│   ├── vlm/
│   └── fusion/
├── data/raw/
├── data/processed/
├── outputs/
├── models/
├── schemas/
├── tests/
├── docs/
├── requirements.txt
├── README_MEMBER1.md
└── README.md
~~~~

## 11. Setup and reproduction

~~~~powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
$env:PYTHONPATH = "$((Get-Location).Path)\src;$env:PYTHONPATH"
~~~~

Download data:

~~~~powershell
python scripts\data\download_datasets.py --skip-sid
~~~~

Create a small CIFAKE experiment:

~~~~powershell
python scripts\data\make_experiment_manifests.py
~~~~

Train:

~~~~powershell
python -u scripts\convnext\train.py --config configs\model\convnext_tiny.yaml --manifest data\processed\train_manifest_small.csv --val-manifest data\processed\val_manifest_small.csv --output-dir outputs\member1\exp02_unfreeze2
~~~~

Predict a directory:

~~~~powershell
python -u scripts\convnext\predict.py --checkpoint outputs\member1\exp02_unfreeze2\best.pt --input-dir data\samples --output-json outputs\member1\predictions.json
~~~~

Run the UI:

~~~~powershell
python -m streamlit run apps\demo\app.py
~~~~

## 12. Expected deliverables

### Public repository

The repository contains:

- structured and commented source code;
- setup instructions;
- dataset and manifest scripts;
- ConvNeXt training and prediction scripts;
- a directory inference script;
- JSON output containing image_path and pred;
- Grad-CAM tooling;
- experiment documentation;
- limitations;
- team ownership and integration contracts.

### Demo video

The demo should show:

1. uploading an image;
2. provenance/forensic processing;
3. ConvNeXt prediction;
4. VLM or fusion output when available;
5. the evidence heatmap; and
6. the final user-facing result.

### Team contributions

| Member | Responsibility |
|---|---|
| Member 1 - Zowie | ConvNeXt, augmentations, training, evaluation, Grad-CAM |
| Member 2 - Jing Eng | C2PA, provenance, provider checks, watermark signals |
| Member 3 - Yong Lin | VLM adapters, prompts, semantic evidence, source estimate |
| Member 4 - Sania | Fusion, orchestration, UI/API integration |
| Member 5 - Aadithiya | Dataset preparation, manifests, evaluation, reporting |

Replace the member numbers with names before submission.

## 13. Limitations

- CIFAKE and SID-Set may contain dataset-specific shortcuts.
- Validation accuracy does not guarantee cross-dataset generalisation.
- Authentic images with unusual processing may be falsely flagged.
- Unfamiliar generators may produce false negatives.
- VLMs can hallucinate explanations.
- API calls add latency and can be rate-limited.
- C2PA may be stripped during reposting.
- Blind forensics can produce false positives on compression and natural textures.
- The ConvNeXt threshold requires calibration.
- Grad-CAM has limited spatial precision.
- The current prototype is not production moderation infrastructure.

## 14. Future work

Given more time, the team would:

- add larger and more diverse training data;
- test SID-Set and other generators separately;
- evaluate clean and transformed images using matched splits;
- calibrate ConvNeXt and fusion probabilities;
- optimize inference latency;
- add automated threshold selection;
- improve UI review workflows;
- validate on WildFake only after model selection;
- investigate dataset shortcuts using cross-dataset testing; and
- fit a held-out fusion model instead of using an interim weighted policy.
