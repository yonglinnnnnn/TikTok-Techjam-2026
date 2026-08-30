# TrueSight - Member 1: ConvNeXt Vision Engine

This module is the **computer-vision / deep-learning component** of TrueSight for the TikTok TechJam 2026 problem **Robust Detection of AI-Generated Images Under Real-World Transformations**.

It intentionally implements **only Member 1's scope**:

- ConvNeXt-Tiny transfer learning
- Robust image augmentation with Albumentations
- staged fine-tuning / selective unfreezing
- optional two-view consistency regularisation
- class-imbalance handling
- mixed-precision training when supported
- validation metrics and checkpointing
- single-image / directory inference
- Grad-CAM heatmaps
- typed integration contracts for future Member 2/3 outputs, without implementing those modules

The official problem statement requires image-level AIGC detection, robustness to JPEG compression, blur, resize, noise, color adjustment and cropping, and models under 2B parameters. The proposed backbone is ConvNeXt-Tiny, which is far below that limit. The problem statement also explicitly identifies ConvNeXt, Albumentations-style transformations and Grad-CAM as appropriate directions for this track. See the source material in the repository / challenge brief.

## Why Python modules instead of a single notebook?

Use Python modules for the shared implementation and optionally use a notebook only for experiments. This avoids merge conflicts between five team members and lets Member 4 import the model directly into the end-to-end pipeline.

Recommended workflow:

```text
Member 5 dataset manifests
          |
          v
Member 1 train.py --> best.pt --> Member 4 pipeline
          |
          +--> predict.py --> image_path + pred JSON
          |
          +--> Grad-CAM --> heatmap PNG

Member 2 provenance ------------------+
                                      |
Member 3 VLM -------------------------+--> Member 4 fusion/orchestration
```

## Dataset contract

Member 1 does **not** hard-code the supplied datasets. Member 5 should prepare a manifest with at least:

```csv
image_path,label,source,split
/path/to/img001.jpg,0,COCO,train
/path/to/img002.jpg,1,CIFAKE,train
```

Labels:

- `0` = authentic / real
- `1` = AI-generated

For the official demonstration benchmark, the WildFake subset must remain validation-only and must never enter the training manifest. The challenge brief explicitly states that the WildFake subset is for demonstration/reference validation and must not be used during training.

## Install

From the repository root:

```bash
pip install -r requirements.txt
```

For a CUDA Colab runtime, install the matching PyTorch build first if necessary, then install the remaining dependencies.

## Quick start

```bash
python scripts/ConvNext/train.py \
  --manifest data/processed/train_manifest.csv \
  --val-manifest data/processed/val_manifest.csv \
  --output-dir outputs/member1/convnext_tiny
```

Then:

```bash
python scripts/ConvNext/predict.py \
  --checkpoint outputs/member1/convnext_tiny/best.pt \
  --input-dir data/samples \
  --output-json outputs/member1/predictions.json
```

Generate a heatmap:

```bash
python scripts/ConvNext/gradcam.py \
  --checkpoint outputs/member1/convnext_tiny/best.pt \
  --image data/samples/example.jpg \
  --output outputs/member1/heatmaps/example.png
```

## Training strategy

### Stage A - classifier warm-up

Freeze the ConvNeXt feature extractor and train only the binary classifier head. This quickly adapts the ImageNet representation to the real-vs-AIGC task without destroying useful pretrained features.

### Stage B - selective fine-tuning

Unfreeze the last 1-3 ConvNeXt stages and continue training with a smaller learning rate. The default is **2 stages**. This is the main transfer-learning parameter to tune.

### Stage C - robustness-aware training

The training loader can produce:

1. a standard training view, and
2. a strongly transformed view of the same image.

The supervised loss is applied to both views. An optional consistency term encourages the model to give similar probabilities to the two views:

```text
L = L_BCE(clean) + L_BCE(transformed) + lambda * L_consistency
```

This is deliberately kept optional because it costs approximately another forward pass and should only be enabled if validation results justify the extra compute.

## Robustness transformations

The default Albumentations pipeline covers the transformations listed by the challenge:

| Transformation | Default training range | Real-world motivation |
|---|---:|---|
| JPEG | quality 30-90 | social-media re-encoding |
| Gaussian blur | sigma 0.5-2.0 | focus / resampling blur |
| Downscale | 0.25x-0.5x | thumbnail / low-resolution repost |
| Gaussian noise | sigma 0.02-0.10 | sensor / transmission noise |
| Color jitter | brightness/contrast/saturation 0.8-1.2 | filters / auto-enhancement |
| Center crop | 80% | profile-picture / framing changes |

Not every transform is applied on every image. Probabilities are configurable so the model still sees enough clean evidence to learn the underlying AIGC signal.

## Parameter tuning priority

Do **not** brute-force a large grid during the hackathon. Use a small, staged search:

1. `unfreeze_stages`: `[1, 2, 3]`
2. head learning rate: `1e-4` to `5e-4`
3. backbone learning rate: `1e-6` to `5e-5`
4. image size: `224` vs `256`
5. augmentation probability / strength
6. consistency weight: `0.0`, `0.02`, `0.05`, `0.1`
7. dropout: `0.0` to `0.3`
8. weight decay: `1e-5` to `1e-3`

Recommended method:

- first run 3-5 cheap experiments with reduced epochs;
- eliminate clearly bad configurations;
- run 2-3 focused experiments around the best configuration;
- retrain the best configuration with full training budget;
- evaluate on WildFake only after model selection is frozen.

This avoids overfitting the public validation benchmark through excessive manual tuning.

## Important evaluation discipline

Do not use WildFake images to tune augmentations, thresholds, or architecture after repeatedly inspecting their results. The challenge says WildFake is a validation/reference dataset and must not be used for training. Treat it as a final external check after the training recipe has been selected.

## Integration contract for Members 2 and 3

Member 1 outputs only the learned visual signal:

```json
{
  "image_path": "...",
  "pred": 0.892
}
```

The eventual pipeline may attach additional fields such as provenance or VLM evidence. This repository deliberately does **not** implement or make decisions from those fields.

Suggested future combined object:

```json
{
  "image_path": "...",
  "pred": 0.892,
  "metadata": {
    "tier1_provenance": null,
    "vlm": null,
    "heatmap_path": "..."
  }
}
```

## What to show judges

For the technical story, Member 1 should be able to demonstrate:

- transfer learning rather than training from scratch;
- selective unfreezing;
- robustness augmentation;
- clean vs transformed evaluation;
- a reproducible checkpoint;
- Grad-CAM showing the spatial evidence used by the classifier;
- a clear latency/accuracy trade-off;
- failure cases and false positives/negatives.

Avoid claiming that Grad-CAM measures the exact percentage of an image that was generated. It is a visual explanation of the classifier's influential regions, not pixel-level ground truth segmentation.
