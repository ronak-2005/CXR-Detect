import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml
from sklearn.metrics import roc_auc_score
from torch.cuda.amp import GradScaler, autocast

from dataset import build_loaders
from model import build_model, save_checkpoint, export_onnx


def get_optimizer(model, cfg):
    head_params     = [p for n, p in model.named_parameters() if "head" in n and p.requires_grad]
    backbone_params = [p for n, p in model.named_parameters() if "backbone" in n and p.requires_grad]

    return torch.optim.AdamW([
        {"params": head_params,     "lr": cfg["training"]["lr"]},
        {"params": backbone_params, "lr": cfg["training"]["lr"] * 0.1},
    ], weight_decay=float(cfg["training"]["weight_decay"]))


def get_scheduler(optimizer, cfg, steps_per_epoch):
    lr           = float(cfg["training"]["lr"])
    min_lr       = float(cfg["scheduler"]["min_lr"])
    warmup_steps = cfg["training"]["warmup_epochs"] * steps_per_epoch
    total_steps  = cfg["training"]["epochs"] * steps_per_epoch

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(
            min_lr/lr,
            0.5 * (1 + np.cos(np.pi * progress))
        )

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


class EarlyStopping:
    def __init__(self, patience):
        self.patience    = patience
        self.best_score  = None
        self.counter     = 0
        self.should_stop = False

    def step(self, score):
        if self.best_score is None or score > self.best_score + 1e-4:
            self.best_score = score
            self.counter    = 0
            return True
        self.counter += 1
        if self.counter >= self.patience:
            self.should_stop = True
        return False


def run_epoch(model, loader, criterion, device, optimizer=None, scheduler=None, scaler=None):
    training = optimizer is not None
    model.train() if training else model.eval()

    total_loss       = 0.0
    all_probs        = []
    all_labels       = []

    for imgs, labels, _ in loader:
        imgs, labels = imgs.to(device), labels.to(device)

        with autocast(enabled=scaler is not None):
            logits = model(imgs)
            loss   = criterion(logits, labels)

        if training:
            optimizer.zero_grad()
            if scaler:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            if scheduler:
                scheduler.step()

        probs = torch.softmax(logits.detach(), dim=1)[:, 1]
        all_probs.extend(probs.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        total_loss += loss.item() * len(labels)

    auc = roc_auc_score(all_labels, all_probs)
    return total_loss / len(all_labels), auc


def train(cfg_path):
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    Path(cfg["paths"]["checkpoints"]).mkdir(parents=True, exist_ok=True)
    Path(cfg["paths"]["outputs"]).mkdir(parents=True, exist_ok=True)

    train_loader, val_loader, _, train_ds, _, _ = build_loaders(cfg)

    model = build_model(cfg, device)
    print(f"Trainable params: {model.trainable_params():,} / {model.total_params():,}")

    counts        = train_ds.class_counts()
    total         = sum(counts)
    class_weights = torch.tensor(
        [total / (len(counts) * c) for c in counts], dtype=torch.float
    ).to(device)

    criterion = nn.CrossEntropyLoss(
        weight=class_weights,
        label_smoothing=cfg["training"]["label_smoothing"]
    )

    optimizer     = get_optimizer(model, cfg)
    scheduler     = get_scheduler(optimizer, cfg, len(train_loader))
    scaler        = GradScaler() if cfg["training"]["mixed_precision"] and device == "cuda" else None
    early_stopper = EarlyStopping(patience=cfg["training"]["early_stopping_patience"])

    history        = {"train_loss": [], "val_loss": [], "train_auc": [], "val_auc": []}
    unfreeze_done  = False

    for epoch in range(1, cfg["training"]["epochs"] + 1):

        if epoch == cfg["training"]["unfreeze_epoch"] and not unfreeze_done:
            model.unfreeze(["layer3", "layer4"])
            optimizer     = get_optimizer(model, cfg)
            scheduler     = get_scheduler(optimizer, cfg, len(train_loader))
            unfreeze_done = True
            print(f"Epoch {epoch}: backbone layer3 + layer4 unfrozen")

        t0 = time.time()
        tr_loss, tr_auc = run_epoch(model, train_loader, criterion, device, optimizer, scheduler, scaler)
        vl_loss, vl_auc = run_epoch(model, val_loader,   criterion, device)
        elapsed         = time.time() - t0

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(vl_loss)
        history["train_auc"].append(tr_auc)
        history["val_auc"].append(vl_auc)

        is_best = early_stopper.step(vl_auc)
        save_checkpoint(
            model, optimizer, epoch, vl_auc,
            f"{cfg['paths']['checkpoints']}/epoch_{epoch:03d}.pt",
            is_best=is_best
        )

        print(
            f"Epoch {epoch:03d}/{cfg['training']['epochs']}  "
            f"train_loss={tr_loss:.4f}  train_auc={tr_auc:.4f}  "
            f"val_loss={vl_loss:.4f}  val_auc={vl_auc:.4f}  "
            f"[{elapsed:.0f}s]  {'best' if is_best else ''}"
        )

        if early_stopper.should_stop:
            print(f"Early stopping at epoch {epoch}")
            break

    with open(cfg["paths"]["history"], "w") as f:
        json.dump(history, f, indent=2)

    export_onnx(model, cfg["paths"]["onnx_model"], cfg["data"]["img_size"], device)
    print("Training complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    train(args.config)