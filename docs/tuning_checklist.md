# Member 1 tuning checklist

## Highest-value knobs

| Priority | Parameter | Suggested values | Why |
|---|---|---|---|
| 1 | `unfreeze_stages` | 1 / 2 / 3 | controls transfer-learning capacity |
| 2 | `backbone_lr` | 1e-5 / 3e-5 / 5e-5 | controls adaptation speed of pretrained features |
| 3 | `head_lr` | 1e-4 / 2e-4 / 5e-4 | controls classifier adaptation |
| 4 | `consistency_weight` | 0 / 0.02 / 0.05 / 0.1 | robustness vs compute |
| 5 | image size | 224 / 256 | detail vs throughput |
| 6 | corruption probabilities | 0.2-0.6 | robustness strength |
| 7 | dropout | 0.0 / 0.15 / 0.3 | overfitting control |
| 8 | weight decay | 1e-5 / 1e-4 / 1e-3 | regularisation |

## Efficient search

Use a **funnel**, not a full Cartesian grid.

```text
5 short trials
   -> keep top 2
2 focused LR trials each
   -> keep top 1
2 robustness trials
   -> final full run
```

A random/Bayesian search library can be added later, but it is not necessary for the first hackathon iteration. The expensive dimension is model training, so manual successive-halving style screening is easier to reason about and log.
