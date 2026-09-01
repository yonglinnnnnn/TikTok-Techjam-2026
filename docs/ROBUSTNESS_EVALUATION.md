# Robustness Evaluation Summary

## Purpose

The challenge requires performance to remain useful after realistic image transformations. The evaluation compares the ConvNeXt model on clean images and transformed versions of the same images.

Transformations include:

- JPEG compression;
- Gaussian blur;
- downscale and upscale;
- Gaussian noise;
- color adjustment; and
- center cropping.

## Evaluation protocol

Use the same labelled images for each condition. Do not change the model checkpoint between conditions.

~~~~text
same validation images
  -> clean evaluation
  -> JPEG evaluation
  -> blur evaluation
  -> resize evaluation
  -> noise evaluation
  -> color evaluation
  -> crop evaluation
~~~~

Report:

- accuracy;
- balanced accuracy;
- precision;
- recall;
- F1-score;
- ROC-AUC;
- false positives;
- false negatives; and
- inference time.

The external WildFake subset must remain a final validation/reference evaluation and must not be used to tune the model.

## Current clean CIFAKE result

The current Run 2 model used:

~~~~text
ConvNeXt-Tiny
pretrained ImageNet weights
last two stages fine-tuned after warm-up
224 x 224 inputs
augmentation disabled
consistency disabled
~~~~

Validation set:

~~~~text
2,000 real images
2,000 AI images
4,000 images total
~~~~

| Condition | Images | Accuracy | Balanced accuracy | F1 | ROC-AUC | Status |
|---|---:|---:|---:|---:|---:|---|
| Clean CIFAKE validation | 4,000 | 95.475% | 95.475% | 95.449% | approximately 99.20% | Measured |
| JPEG | 4,000 | Pending | Pending | Pending | Pending | Run required |
| Blur | 4,000 | Pending | Pending | Pending | Pending | Run required |
| Resize | 4,000 | Pending | Pending | Pending | Pending | Run required |
| Noise | 4,000 | Pending | Pending | Pending | Pending | Run required |
| Color adjustment | 4,000 | Pending | Pending | Pending | Pending | Run required |
| Center crop | 4,000 | Pending | Pending | Pending | Pending | Run required |

Do not replace Pending with estimated values. The transformed rows must be measured.

## Interpretation

The clean result shows strong performance on this CIFAKE validation subset. It does not establish robustness by itself.

The robustness claim is supported only if transformed-set metrics are measured and compared against the clean baseline.

The most important comparison is the performance drop:

~~~~text
clean metric - transformed metric
~~~~

A small drop suggests useful robustness. A large drop identifies a transformation for further augmentation or consistency-training experiments.

## Reproduction checklist

1. Freeze the checkpoint.
2. Use a fixed validation manifest.
3. Generate one transformed version per condition.
4. Keep labels unchanged.
5. Run the same evaluation code for every condition.
6. Save each report with the transformation name.
7. Record the transformation parameters.
8. Compare false positives and false negatives.
9. Do not tune repeatedly on WildFake.
10. Include the final table in the submission.
