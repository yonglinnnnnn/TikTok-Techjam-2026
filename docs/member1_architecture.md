# Member 1 Architecture and Training Notes

## 1. Scope boundary

Member 1 owns the learned visual detector and explainability layer. Member 1 does not implement C2PA, SynthID, OpenAI, Gemini, Anthropic, final score fusion, Streamlit/Gradio, or benchmark orchestration.

## 2. Model

```text
RGB image
   |
   v
Albumentations robustness pipeline
   |
   v
224x224 tensor + ImageNet normalization
   |
   v
ConvNeXt-Tiny pretrained backbone
   |
   +-- frozen early representation
   |
   +-- last N stages selectively fine-tuned
   |
   v
Global average pooling / pretrained ConvNeXt classifier path
   |
   v
Dropout
   |
   v
1 logit
   |
   v
Sigmoid -> P(AIGC)
```

The backbone is intentionally kept well below the hackathon's `<2B parameters` limit.

## 3. Why ConvNeXt-Tiny

The challenge asks for robust image-level AIGC detection under real-world transformations and explicitly allows public pretrained weights. ConvNeXt-Tiny gives a strong modern convolutional representation while remaining practical on a Colab GPU and easy to deploy in a lightweight inference path.

The detector should learn generator-related visual evidence without requiring a giant foundation model.

## 4. Transfer learning

The classifier head is always trainable. The default configuration also unfreezes the final two ConvNeXt stages.

The main tunable parameter is:

```yaml
model:
  unfreeze_stages: 2
```

Try:

- `1`: safest / fastest; less adaptation
- `2`: recommended starting point
- `3`: more adaptation; higher overfitting risk

Do not start with full-network fine-tuning. The hackathon timeline rewards a reliable experiment loop more than an expensive search.

## 5. Differential learning rates

The pretrained backbone uses a smaller learning rate than the new classifier:

```yaml
backbone_lr: 0.00001
head_lr: 0.0002
```

This protects pretrained low-level representations while allowing the new task-specific head to adapt quickly.

## 6. Robustness-aware augmentation

Training uses a mixture of corruption families identified by the problem statement:

- JPEG quality 30-90
- Gaussian blur sigma 0.5-2.0
- downscale 0.25x-0.5x followed by upscale
- Gaussian noise 0.02-0.10
- color jitter approximately +/-20%
- 80% center crop

The important design choice is **stochastic corruption rather than permanently corrupting the training data**. The model sees both clean-ish and degraded examples and is less likely to treat one fixed corruption pattern as the definition of AIGC.

## 7. Optional consistency regularisation

When enabled, the loader creates a second transformed view of the same source image. The loss is:

```text
BCE(clean, label)
+ BCE(transformed, label)
+ lambda * MSE(P(clean), P(transformed))
```

The implementation detaches the clean probability in the consistency term, making the transformed branch follow the clean branch rather than creating a potentially unstable two-way objective.

Tune:

```yaml
training:
  consistency_weight: 0.0
  # then 0.02, 0.05, 0.1
```

If it slows training substantially without improving transformed-set performance, set it to `0.0`.

## 8. Grad-CAM

Grad-CAM uses the last ConvNeXt feature block's depthwise convolution. It computes channel weights from the gradient of the AIGC logit and projects them back to image space.

Interpretation:

- bright region = region that contributed strongly to the AIGC logit
- dark region = relatively weak contribution

It is **not** a pixel-level segmentation model. Therefore do not report the heatmap as an exact `% generated` measurement.

## 9. Model selection

The training script records:

- validation loss
- ROC-AUC
- average precision
- F1
- balanced accuracy
- accuracy
- training/validation runtime
- learning rates
- consistency loss

Balanced accuracy is the default local checkpoint-selection metric because a simple 0.5 threshold can hide class imbalance. For the hackathon report, Member 5 should also provide the required clean-vs-transformed robustness table.

## 10. Efficient tuning strategy

### Phase 1: transfer-learning depth

Run short experiments for:

```text
unfreeze_stages = 1, 2, 3
```

### Phase 2: learning rate

Around the best stage depth:

```text
backbone_lr = 1e-5, 3e-5, 5e-5
head_lr     = 1e-4, 2e-4, 5e-4
```

### Phase 3: robustness strength

Compare:

```text
baseline augmentation
strong augmentation
strong augmentation + consistency
```

### Phase 4: final full run

Freeze the recipe and train with the final number of epochs. Only after that should the team perform the official WildFake demonstration evaluation.

## 11. Recommended experiment log columns

```text
run_id
commit
unfreeze_stages
backbone_lr
head_lr
image_size
augmentation_profile
consistency_weight
epochs
best_epoch
val_balanced_accuracy
val_roc_auc
val_average_precision
val_f1
clean_accuracy
transformed_accuracy
runtime_seconds
notes
```

This lets Member 5 create the robustness report without modifying Member 1's code.
