from __future__ import annotations

import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from .checkpoint import save_checkpoint, save_json
from .config import Config
from .dataset import AIGCManifestDataset
from .losses import BinaryAIGCLoss
from .metrics import binary_metrics
from .model import ConvNeXtAIGCDetector


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_loaders(config: Config, train_manifest: str, val_manifest: str):
    train_ds = AIGCManifestDataset(
        train_manifest,
        image_size=config.model.target_image_size,
        train=True,
        consistency_view=config.training.use_consistency_view,
        augmentation_config=config.augmentation,
    )
    val_ds = AIGCManifestDataset(
        val_manifest,
        image_size=config.model.target_image_size,
        train=False,
        consistency_view=False,
        augmentation_config=config.augmentation,
    )

    common = {
        "batch_size": config.training.batch_size,
        "num_workers": config.training.num_workers,
        "pin_memory": config.training.pin_memory and torch.cuda.is_available(),
    }
    if config.training.num_workers > 0:
        common["persistent_workers"] = config.training.persistent_workers

    train_loader = DataLoader(train_ds, shuffle=True, drop_last=False, **common)
    val_loader = DataLoader(val_ds, shuffle=False, drop_last=False, **common)
    return train_loader, val_loader


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _autocast(device: torch.device, enabled: bool):
    if device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.float16, enabled=enabled)
    return torch.autocast(device_type="cpu", dtype=torch.bfloat16, enabled=False)


def _train_one_epoch(model, loader, criterion, optimizer, scaler, device, config):
    model.train()
    losses = []
    supervised_losses = []
    consistency_losses = []

    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)
        consistency_images = batch.get("image_consistency")
        if consistency_images is not None:
            consistency_images = consistency_images.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        use_amp = bool(config.training.amp and device.type == "cuda")
        with _autocast(device, use_amp):
            logits = model(images)
            consistency_logits = model(consistency_images) if consistency_images is not None else None
            loss, pieces = criterion(logits, labels, consistency_logits)

        scaler.scale(loss).backward() if use_amp else loss.backward()
        if config.training.grad_clip_norm > 0:
            if use_amp:
                scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), config.training.grad_clip_norm)
        if use_amp:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()

        losses.append(pieces["total"])
        supervised_losses.append(pieces["supervised"])
        consistency_losses.append(pieces["consistency"])

    return {
        "loss": float(np.mean(losses)),
        "supervised_loss": float(np.mean(supervised_losses)),
        "consistency_loss": float(np.mean(consistency_losses)),
    }


@torch.no_grad()
def _validate(model, loader, device):
    model.eval()
    labels = []
    probabilities = []
    losses = []
    criterion = nn.BCEWithLogitsLoss()

    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        targets = batch["label"].to(device, non_blocking=True)
        logits = model(images)
        losses.append(float(criterion(logits, targets).item()))
        probabilities.extend(torch.sigmoid(logits).cpu().numpy().tolist())
        labels.extend(targets.cpu().numpy().tolist())

    metrics = binary_metrics(labels, probabilities)
    metrics["loss"] = float(np.mean(losses))
    return metrics


def train(config: Config, train_manifest: str, val_manifest: str, output_dir: str | Path) -> dict:
    seed_everything(config.seed)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = _device()
    train_loader, val_loader = build_loaders(config, train_manifest, val_manifest)

    model = ConvNeXtAIGCDetector(
        pretrained=config.model.pretrained,
        dropout=config.model.dropout,
        unfreeze_stages=config.model.unfreeze_stages,
    ).to(device)

    backbone_parameters = []
    head_parameters = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if name.startswith("backbone.classifier"):
            head_parameters.append(parameter)
        else:
            backbone_parameters.append(parameter)

    # Keep the selected stages in the optimizer, but freeze them during the
    # classifier warm-up so they can be enabled without rebuilding optimizer
    # or scheduler state.
    if config.training.warmup_epochs > 0:
        for parameter in backbone_parameters:
            parameter.requires_grad = False

    optimizer = AdamW(
        [
            {"params": backbone_parameters, "lr": config.training.backbone_lr},
            {"params": head_parameters, "lr": config.training.head_lr},
        ],
        weight_decay=config.training.weight_decay,
    )

    report = model.freeze_report()
    
    optimizer = AdamW(
        [
            {"params": backbone_parameters, "lr": config.training.backbone_lr},
            {"params": head_parameters, "lr": config.training.head_lr},
        ],
        weight_decay=config.training.weight_decay,
    )

    optimizer_parameter_ids = {
        id(parameter)
        for group in optimizer.param_groups
        for parameter in group["params"]
    }

    missing_parameters = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad
        and id(parameter) not in optimizer_parameter_ids
    ]

    print(
        "After warm-up: "
        f"trainable={report.trainable_parameters:,} / "
        f"{report.total_parameters:,}",
        flush=True,
    )

    print(
        "Optimizer missing trainable parameters: "
        f"{len(missing_parameters):,}",
        flush=True,
    )

    print(
        f"Device={device} | trainable={report.trainable_parameters:,} "
        f"/ {report.total_parameters:,} ({100 * report.trainable_ratio:.2f}%)"
    )

    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=max(config.training.epochs, 1),
        eta_min=config.training.backbone_lr * config.scheduler.min_lr_ratio,
    )

    criterion = BinaryAIGCLoss(
        consistency_weight=config.training.consistency_weight if config.training.use_consistency_view else 0.0,
        label_smoothing=config.training.label_smoothing,
    )
    use_amp = bool(config.training.amp and device.type == "cuda")
    scaler = torch.cuda.amp.GradScaler("cuda", enabled=use_amp)

    history = []
    best_score = -math.inf
    best_epoch = -1
    patience = 0
    started = time.perf_counter()

    save_json(output_dir / "config.json", {
        "seed": config.seed,
        "model": vars(config.model),
        "training": vars(config.training),
        "augmentation": vars(config.augmentation),
        "scheduler": vars(config.scheduler),
    })

    for epoch in range(1, config.training.epochs + 1):
        if epoch == config.training.warmup_epochs + 1 and backbone_parameters:
            for parameter in backbone_parameters:
                parameter.requires_grad = True
            print(f"Warm-up complete; unfroze the last {config.model.unfreeze_stages} ConvNeXt stage(s)")
        epoch_start = time.perf_counter()
        train_metrics = _train_one_epoch(model, train_loader, criterion, optimizer, scaler, device, config)
        val_metrics = _validate(model, val_loader, device)
        scheduler.step()

        # Balanced accuracy is used as the default model-selection score because
        # the external validation set can be class-imbalanced. AUC is also logged.
        selection_score = val_metrics["balanced_accuracy"]
        record = {
            "epoch": epoch,
            "train": train_metrics,
            "val": val_metrics,
            "selection_score": selection_score,
            "lr": [group["lr"] for group in optimizer.param_groups],
            "epoch_seconds": time.perf_counter() - epoch_start,
        }
        history.append(record)
        print(json.dumps(record))

        checkpoint_kwargs = {
            "model_config": {
                "dropout": config.model.dropout,
                "unfreeze_stages": config.model.unfreeze_stages,
            }
        }
        if config.output.save_last:
            save_checkpoint(output_dir / "last.pt", model, optimizer, scheduler, epoch, val_metrics, **checkpoint_kwargs)

        if selection_score > best_score:
            best_score = selection_score
            best_epoch = epoch
            patience = 0
            if config.output.save_best:
                save_checkpoint(output_dir / "best.pt", model, optimizer, scheduler, epoch, val_metrics, **checkpoint_kwargs)
        else:
            patience += 1

        if patience >= config.training.early_stopping_patience:
            print(f"Early stopping at epoch {epoch}; best epoch={best_epoch}")
            break

    elapsed = time.perf_counter() - started
    summary = {
        "best_epoch": best_epoch,
        "best_selection_score": best_score,
        "elapsed_seconds": elapsed,
        "device": str(device),
        "history": history,
    }
    save_json(output_dir / "training_history.json", summary)
    return summary
