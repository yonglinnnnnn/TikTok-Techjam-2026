# Member 1 -> Team Integration Contract

## Owned by Member 1

```json
{
  "image_path": "data/val/img_001.jpg",
  "pred": 0.892
}
```

`pred` is the model's estimated probability that the image is AI-generated.

## Optional explanation artifact

```text
outputs/heatmaps/<image_stem>.png
```

## Not owned by Member 1

The following fields are reserved for later integration:

```json
{
  "metadata": {
    "tier1_provenance": null,
    "vlm": null
  }
}
```

Member 1's model must not inspect or fuse these values. Member 4 owns the final orchestration and score-fusion policy.
