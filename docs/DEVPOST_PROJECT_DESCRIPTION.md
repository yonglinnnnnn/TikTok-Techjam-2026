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
