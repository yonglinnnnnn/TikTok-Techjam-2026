# TrueSight — ConvNeXt Vision Engine

This module detects whether an image is likely to be:

- authentic / real; or
- AI-generated or AI-manipulated.

It is designed for the TikTok TechJam challenge:

> Robust Detection of AI-Generated Images Under Real-World Transformations

The model focuses on visual evidence from image pixels. It does not depend on C2PA metadata, watermarks, provenance APIs, or VLM outputs. Those signals can be combined later by Member 4 through the project fusion layer.

---

## 1. Responsibilities

Member 1 implements:

- ConvNeXt-Tiny transfer learning;
- binary real-versus-AI classification;
- selective ConvNeXt fine-tuning;
- robustness-oriented image augmentation;
- optional consistency training;
- validation metrics;
- model checkpointing;
- batch image prediction;
- Grad-CAM explanations;
- a clean interface for Member 4.

Member 1 does not implement:

- the Streamlit or Gradio interface;
- C2PA verification;
- SynthID or watermark detection;
- OpenAI, Gemini, or Anthropic VLM analysis;
- final score fusion;
- final WildFake benchmarking;
- platform-wide moderation logic.

The model produces an independent visual probability:

    P(AIGC | image)

where:

- 0.0 = model considers the image likely authentic
- 1.0 = model considers the image likely AI-generated

This value is a model score, not a guaranteed probability of authorship.

## 2. Why ConvNeXt-Tiny?

ConvNeXt is a modern convolutional neural network designed to achieve strong image-recognition performance while retaining the practical advantages of convolutional models.

ConvNeXt-Tiny is suitable for this hackathon because:

- it is substantially below the two-billion-parameter limit;
- it has publicly available ImageNet pretrained weights;
- it provides strong visual features without training from scratch;
- it can be fine-tuned on limited hackathon hardware;
- it is suitable for image-level classification;
- its convolutional feature maps can be used for Grad-CAM explanations.

The original TorchVision ConvNeXt-Tiny contains approximately 28.6 million parameters. After replacing the 1,000-class ImageNet classifier with a one-logit binary classifier, this project uses approximately 27.8 million parameters.

The model is therefore small enough for a practical prototype while still providing a strong visual backbone.

## 3. Model architecture

The model performs the following operation:

    RGB image
        ↓
    Resize and crop
        ↓
    Optional robustness transformation
        ↓
    ImageNet normalization
        ↓
    Pretrained ConvNeXt-Tiny backbone
        ↓
    Global feature representation
        ↓
    Dropout
        ↓
    Linear layer with one output logit
        ↓
    Sigmoid
        ↓
    P(AIGC)

The classifier output is one logit rather than two class logits:

    logit = model(image)
    pred = sigmoid(logit)

The project uses the following binary labels:

- 0 = real/authentic
- 1 = AI-generated or AI-manipulated

## 4. Transfer learning strategy

Training the entire ConvNeXt model from random initialization would require substantially more data and compute.
Instead, this project starts with ImageNet pretrained weights.
The training process has two stages.

### Stage A: classifier warm-up
During the warm-up period:

- ConvNeXt feature extractor: frozen
- Binary classifier head: trainable

This allows the new classifier to quickly adapt to the real-versus-AI task without immediately changing the pretrained visual representation.

### Stage B: selective fine-tuning
After warm-up, the last selected ConvNeXt stages are unfrozen.
The configuration controls this:

    model:
      unfreeze_stages: 2

The supported values are:
- 0 = classifier only
- 1 = last ConvNeXt stage
- 2 = last two ConvNeXt stages
- 3 = last three ConvNeXt stages
- 4 = all four ConvNeXt stages

The recommended starting value is:
    
    unfreeze_stages: 2

The selected backbone stages use a smaller learning rate than the new classifier head:

    training:
      backbone_lr: 0.00001
      head_lr: 0.0002

This protects useful pretrained features while allowing the classifier to adapt more quickly.

### Trainable-parameter interpretation
During warm-up, a run may print approximately:
    
    trainable = 2,305

This means only the binary classifier is currently trainable.

After warm-up, with `unfreeze_stages: 2`, approximately 25 million parameters become trainable. This is expected and confirms that the last two ConvNeXt stages have been enabled.

## 5. Robustness objective

The challenge requires models to remain useful after realistic image redistribution operations, including:

- JPEG compression;
- Gaussian blur;
- resizing and upscaling;
- Gaussian noise;
- brightness and color changes;
- center cropping.

A detector trained only on pristine images may learn fragile details that disappear after an image is uploaded to a social-media platform.
This project therefore exposes the model to controlled transformations during training.

The objective is:
- clean or normally transformed image → correct label
- realistically degraded image → same correct label

The model should learn evidence that remains useful after ordinary reposting and processing.

## 6. Augmentation pipeline

Augmentations are implemented in:
`src/truesight/vision/augmentations.py`

The current pipeline uses Albumentations 2.x.
The main transformations are:

| Transformation | Configuration | Real-world interpretation |
| --- | --- | --- |
| JPEG compression | quality 30–90 | Social-media or messaging re-encoding |
| Gaussian blur | sigma 0.5–2.0 | Out-of-focus or processing blur |
| Downscale/upscale | 0.25–0.5 scale | Thumbnail or low-resolution repost |
| Gaussian noise | standard deviation 0.02–0.10 | Transmission or sensor noise |
| Color jitter | approximately ±20% | Filters and auto-enhancement |
| Center crop | approximately 80% | Profile-picture or reframing crop |
| Horizontal flip | configurable | Natural image variation |

The corruption pipeline does not apply every severe transformation simultaneously. Instead, one corruption family is selected with a configured probability. This prevents the training images from becoming unrealistically damaged.

Example configuration:
    
    augmentation:
      enabled: true
      jpeg_quality: [30, 90]
      blur_sigma: [0.5, 2.0]
      downscale: [0.25, 0.5]
      noise_std: [0.02, 0.10]
      color_jitter: [0.8, 1.2]
      crop_scale: [0.8, 1.0]

The training-time pipeline and the validation-time pipeline are intentionally different:

- **Training:** random crop + optional robustness transformations
- **Validation:** deterministic resize + center crop + normalization

This prevents random augmentation from contaminating validation measurements.

## 7. Consistency training

The project optionally trains using two views of the same image:

    Original image
        ├── normal view
        └── transformed view

The model produces:
- `prediction_original`
- `prediction_transformed`

The training objective is conceptually:

    loss =
        BCE(original, label)
        + BCE(transformed, label)
        + lambda × consistency_loss

The consistency loss penalizes large differences between the two predictions.
For example:
- Original image: P(AI) = 0.92
- Compressed image: P(AI) = 0.88
These predictions are reasonably consistent.

However:
- Original image: P(AI) = 0.92
- Compressed image: P(AI) = 0.31
indicates that the model may be relying on fragile visual evidence.

Consistency training encourages the prediction to remain stable when the image undergoes realistic redistribution transformations. It does not identify exact generated pixels and should not be described as image segmentation.

Configuration:
    
    training:
      consistency_weight: 0.05
      use_consistency_view: true

Begin with:
    
    consistency_weight: 0.0
    use_consistency_view: false

Then compare against:
    
    consistency_weight: 0.02
    use_consistency_view: true
and:
    
    consistency_weight: 0.05
    use_consistency_view: true

Consistency training requires an additional model forward pass and may increase training time.

## 8. Project structure

    truesight_member1/
    │
    ├── configs/
    │   └── model/
    │       └── convnext_tiny.yaml
    │
    ├── scripts/
    │   ├── data/
    │   │   ├── download_datasets.py
    │   │   └── make_experiment_manifests.py
    │   │
    │   └── convnext/
    │       ├── train.py
    │       ├── predict.py
    │       ├── gradcam.py
    │       └── tune.py
    │
    ├── src/
    │   └── truesight/
    │       └── vision/
    │           ├── model.py
    │           ├── dataset.py
    │           ├── augmentations.py
    │           ├── train.py
    │           ├── inference.py
    │           ├── gradcam.py
    │           ├── losses.py
    │           ├── metrics.py
    │           ├── checkpoint.py
    │           ├── config.py
    │           └── integration_contract.py
    │
    ├── data/
    │   ├── raw/
    │   └── processed/
    │
    ├── outputs/
    │   └── member1/
    │
    ├── models/
    │   └── ConvNext/
    │
    ├── docs/
    ├── tests/
    ├── requirements.txt
    └── README_MEMBER1.md

Raw datasets, generated manifests, checkpoints, predictions, and heatmaps should not normally be committed directly to GitHub.

## 9. Dataset contract

Member 1 consumes CSV manifests.
Minimum required columns:

    image_path,label,source,split
    data/image001.jpg,0,CIFAKE,train
    data/image002.jpg,1,CIFAKE,train

Required labels:
- 0 = real/authentic
- 1 = AI-generated or AI-manipulated

The dataset loader converts each image into:

    RGB image
        ↓
    NumPy array
        ↓
    Albumentations transformation
        ↓
    PyTorch tensor
        ↓
    ConvNeXt-Tiny

The model code does not hard-code CIFAKE or SID-Set. This allows the team to change datasets without rewriting the neural-network implementation.
WildFake must remain validation-only and must not appear in training manifests.

## 10. Installation

From the repository root:

    python -m venv .venv
    .venv\Scripts\Activate.ps1
    python -m pip install --upgrade pip
    python -m pip install -r requirements.txt

Set the source path for the current PowerShell session:
    
    $env:PYTHONPATH = "$((Get-Location).Path)\src;$env:PYTHONPATH"

Verify the model package:
    
    python -c "from truesight.vision.model import ConvNeXtAIGCDetector; print('Member 1 import successful')"

## 11. Training

Example baseline:
    
    python -u scripts\convnext\train.py `
        --config configs\model\convnext_tiny.yaml `
        --manifest data\processed\train_manifest.csv `
        --val-manifest data\processed\val_manifest.csv `
        --output-dir outputs\member1\convnext_tiny

The output directory contains:
- `best.pt`
- `last.pt`
- `config.json`
- `training_history.json`

`best.pt` is the checkpoint with the best validation balanced accuracy.

Example experiment:
    
    python -u scripts\convnext\train.py `
        --config configs\model\convnext_tiny.yaml `
        --manifest data\processed\train_manifest_small.csv `
        --val-manifest data\processed\val_manifest_small.csv `
        --output-dir outputs\member1\exp02_unfreeze2

Never overwrite previous experiment directories. Use separate folders for each configuration.

## 12. Evaluation metrics

The validation process records:
- accuracy;
- balanced accuracy;
- precision;
- recall;
- F1-score;
- ROC-AUC;
- average precision;
- validation loss;
- false positives;
- false negatives;
- training time;
- learning rates.

Balanced accuracy is used for checkpoint selection because a raw accuracy score can hide poor performance on an imbalanced validation set. For balanced CIFAKE validation data, accuracy and balanced accuracy may be identical.

A proper evaluation should use a complete validation manifest rather than judging the model from a few manually selected images.

## 13. Prediction interface

The batch prediction script accepts an image directory and writes JSON:

    python -u scripts\convnext\predict.py `
        --checkpoint outputs\member1\exp02_unfreeze2\best.pt `
        --input-dir data\test `
        --output-json outputs\member1\predictions.json

Example output:
    
    [
      {
        "image_path": "data/test/example.jpg",
        "pred": 0.892
      }
    ]

Interpretation:
- pred near 0.0 = likely real
- pred near 1.0 = likely AI-generated

The threshold used by the UI should be configurable. A default threshold of 0.5 is acceptable for initial testing, but the final threshold should be selected using held-out validation data.

## 14. Grad-CAM

Grad-CAM generates a visual explanation of the ConvNeXt decision. It highlights image regions that contributed strongly to the AI-generation score.

Run:
    
    python -u scripts\convnext\gradcam.py `
        --checkpoint outputs\member1\exp02_unfreeze2\best.pt `
        --image data\test\example.jpg `
        --output outputs\member1\heatmaps\example.png

Grad-CAM should be described to judges as:
*A model-evidence heatmap showing regions that contributed to the classifier's AI-generation prediction.*

Do not describe it as:
- the exact percentage of the image generated by AI;
- a pixel-level segmentation mask;
- proof that every highlighted region is synthetic.

Grad-CAM explains the classifier. It does not establish ground-truth manipulation boundaries.

For the UI, use a label such as:
*Regions influencing the AI prediction*

## 15. Recommended experiment sequence

Use a staged experiment plan.

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
- **Experiment 3: robustness training**
  - last 2 stages unfrozen
  - augmentation enabled
  - consistency disabled
- **Experiment 4: consistency training**
  - last 2 stages unfrozen
  - augmentation enabled
  - consistency enabled
  - consistency weight = 0.02 or 0.05

Compare every experiment using the same validation manifest and report:
- accuracy
- balanced accuracy
- F1
- ROC-AUC
- false positives
- false negatives
- training time

The final model should be selected using validation results first, then checked once on the external WildFake reference set.

## 16. Limitations

The model has several important limitations:
- CIFAKE and SID-Set may contain dataset-specific visual shortcuts;
- high validation accuracy does not guarantee cross-dataset generalisation;
- real images with unusual processing may be falsely flagged;
- AI-generated images that resemble the training distribution may be easier to detect;
- threshold 0.5 is not automatically optimal;
- the output is a visual model score, not proof of authorship;
- Grad-CAM is explanatory rather than pixel-accurate;
- CPU training can be slow;
- a model trained only on CIFAKE should not be presented as universally reliable.

A small manually selected folder should not be used as the main evaluation. Use the complete validation manifest and an external dataset whenever possible.
