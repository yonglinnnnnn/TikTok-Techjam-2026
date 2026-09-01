# Error Analysis Note

## Purpose

Error analysis checks where the system fails and whether the failure is caused by:

- dataset shortcuts;
- unusual authentic images;
- unfamiliar AI generation;
- image transformations;
- threshold selection;
- provenance absence; or
- semantic ambiguity.

The aim is not only to report one accuracy number. It is to understand the trade-offs of the proposed approach.

## Current ConvNeXt validation result

The current Run 2 report evaluates 4,000 balanced CIFAKE validation images at threshold 0.5:

~~~~text
2,000 real
2,000 AI
~~~~

Confusion matrix:

| | Predicted real | Predicted AI |
|---|---:|---:|
| Actual real | 1,921 | 79 |
| Actual AI | 102 | 1,898 |

Therefore:

- true negatives / correctly classified real images: 1,921;
- false positives: 79;
- false negatives: 102;
- true positives / correctly classified AI images: 1,898;
- total correct: 3,819 of 4,000;
- accuracy: 95.475%;
- real recall/specificity: 96.05%;
- AI recall/sensitivity: 94.90%;
- AI precision: 96.00%;
- macro F1: approximately 95.45%.

## Main trade-off

At threshold 0.5, the model produces:

~~~~text
79 false positives on real images
102 false negatives on AI images
~~~~

The model misses more AI images than it falsely flags real images. This is a moderate recall trade-off and may matter differently depending on the product objective.

For a safety-oriented detector, the team may prefer higher AI recall. For a platform trying to avoid incorrectly labelling authentic user content, specificity may receive more weight.

The threshold must therefore be selected using held-out validation data rather than assumed to be universally optimal.

## Representative failure cases

### False positives

A false positive is an authentic image predicted as AI-generated.

Potential causes include:

- strong compression;
- unusual camera processing;
- smooth textures;
- image resizing;
- content that resembles the CIFAKE training distribution;
- natural repetitive patterns; or
- a dataset-specific shortcut.

The earlier 10-image Grad-CAM smoke test contained several authentic images with high scores. That folder was too small for a performance estimate, but it demonstrates why false-positive inspection is useful.

Representative files from that smoke test included:

~~~~text
real/0000 (2).jpg
real/0000 (3).jpg
real/0000 (4).jpg
~~~~

These should be reviewed with their heatmaps and compared against the complete validation report.

### False negatives

A false negative is an AI-generated image predicted as real.

Possible causes include:

- a generator not represented in training;
- realistic texture and lighting;
- weak or removed generative artifacts;
- compression that removes useful evidence;
- an image that resembles the authentic training distribution; or
- an overly conservative decision threshold.

False negatives are especially important because a detector can appear accurate while failing on a particular generator family.

### Forensic and provenance disagreement

A missing C2PA record is not evidence that an image is real. C2PA can be removed during editing or reposting.

Similarly:

- a forensic anomaly is not proof of AI generation;
- a Grad-CAM highlight is not a manipulation mask;
- VLM reasoning can be uncertain or hallucinated;
- an unavailable provider API is not a negative result.

The final fusion layer must preserve these distinctions using null, negative, and verified states.

