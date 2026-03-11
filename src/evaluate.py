import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import yaml
from PIL import Image
from sklearn.metrics import (
    roc_auc_score, roc_curve,
    average_precision_score, precision_recall_curve,
    confusion_matrix, classification_report,
    brier_score_loss, f1_score,
    precision_score, recall_score,
)

from dataset import build_loaders, get_transforms,get_tta_transforms, CLASSES
from model import build_model, load_checkpoint
from gradcam import GradCAM, render_overlay, plot_gradcam


def collect_predictions(model, loader, device):
    model.eval()
    all_probs  = []
    all_labels = []
    all_paths  = []

    for imgs, labels, paths in loader:
        imgs = imgs.to(device)
        with torch.no_grad():
            logits = model(imgs)
            probs  = torch.softmax(logits, dim=1)[:, 1]
        all_probs.extend(probs.cpu().numpy())
        all_labels.extend(labels.numpy())
        all_paths.extend(paths)

    return np.array(all_labels), np.array(all_probs), all_paths


def collect_predictions_tta(model, loader, device, cfg):
    model.eval()
    all_probs  = []
    all_labels = []
    all_paths  = []
    img_size   = cfg["data"]["img_size"]
    tta_tfs    = get_tta_transforms(img_size)

    for imgs, labels, paths in loader:
        for i in range(len(imgs)):
            path    = paths[i]
            img_pil = Image.open(path).convert("RGB")
            probs_tta = []
            for tf in tta_tfs:
                t = tf(img_pil).unsqueeze(0).to(device)
                with torch.no_grad():
                    prob = torch.softmax(model(t), dim=1)[0, 1].item()
                probs_tta.append(prob)
            all_probs.append(float(np.mean(probs_tta)))
            all_labels.append(labels[i].item())
            all_paths.append(path)

    return np.array(all_labels), np.array(all_probs), all_paths


def find_best_threshold(labels, probs):
    thresholds = np.linspace(0.1, 0.9, 81)
    best_t, best_f1 = 0.5, 0.0
    for t in thresholds:
        f1 = f1_score(labels, (probs >= t).astype(int), zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_t  = t
    return best_t, best_f1


def plot_roc_pr(labels, probs, save_dir):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    fpr, tpr, _ = roc_curve(labels, probs)
    auc = roc_auc_score(labels, probs)
    axes[0].plot(fpr, tpr, lw=2, label=f"AUC = {auc:.4f}")
    axes[0].plot([0, 1], [0, 1], "k--", lw=1)
    axes[0].set_xlabel("False positive rate")
    axes[0].set_ylabel("True positive rate")
    axes[0].set_title("ROC Curve")
    axes[0].legend()

    prec, rec, _ = precision_recall_curve(labels, probs)
    ap = average_precision_score(labels, probs)
    axes[1].plot(rec, prec, lw=2, label=f"AP = {ap:.4f}")
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].set_title("Precision-Recall Curve")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(save_dir / "roc_pr.png", dpi=150, bbox_inches="tight")
    plt.close()


def plot_confusion(labels, preds, threshold, save_dir):
    cm = confusion_matrix(labels, preds)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=CLASSES, yticklabels=CLASSES, ax=ax)
    ax.set_title(f"Confusion matrix (threshold={threshold:.2f})")
    ax.set_ylabel("Actual")
    ax.set_xlabel("Predicted")
    plt.tight_layout()
    plt.savefig(save_dir / "confusion_matrix.png", dpi=150, bbox_inches="tight")
    plt.close()


def plot_score_distribution(labels, probs, threshold, save_dir):
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.hist(probs[labels == 0], bins=40, alpha=0.7, label="Normal",    color="#2980b9")
    ax.hist(probs[labels == 1], bins=40, alpha=0.7, label="Pneumonia", color="#c0392b")
    ax.axvline(threshold, color="black", ls="--", lw=1.5,
               label=f"Threshold ({threshold:.2f})")
    ax.set_xlabel("Pneumonia probability")
    ax.set_ylabel("Count")
    ax.set_title("Score distribution by class")
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_dir / "score_distribution.png", dpi=150, bbox_inches="tight")
    plt.close()


def error_analysis(model, labels, probs, paths, threshold, cfg, device, save_dir):
    preds      = (probs >= threshold).astype(int)
    wrong_mask = preds != labels
    wrong_idx  = np.where(wrong_mask)[0]

    confidence = np.abs(probs[wrong_mask] - 0.5)
    top_idx    = wrong_idx[np.argsort(confidence)[::-1][:4]]

    img_size = cfg["data"]["img_size"]
    val_tf   = get_transforms(img_size, "val", cfg)
    gradcam  = GradCAM(model, model.get_gradcam_layer())

    fig, axes = plt.subplots(len(top_idx), 3, figsize=(13, 4 * len(top_idx)))
    if len(top_idx) == 1:
        axes = axes[np.newaxis, :]

    for row, idx in enumerate(top_idx):
        img_pil = Image.open(paths[idx]).convert("RGB")
        img_t   = val_tf(img_pil)
        cam, _, _ = gradcam.generate(img_t)
        img_np, _, overlay = render_overlay(img_pil, cam, img_size)

        gt   = CLASSES[labels[idx]]
        pred = CLASSES[preds[idx]]
        prob = probs[idx]

        axes[row, 0].imshow(img_np);          axes[row, 0].set_title(f"GT: {gt}")
        axes[row, 1].imshow(cam, cmap="jet"); axes[row, 1].set_title("Grad-CAM")
        axes[row, 2].imshow(overlay);         axes[row, 2].set_title(f"Pred: {pred} ({prob:.1%})")
        for col in range(3):
            axes[row, col].axis("off")

    plt.tight_layout()
    plt.savefig(save_dir / "error_gradcam.png", dpi=150, bbox_inches="tight")
    plt.close()
    gradcam.remove_hooks()

    return int(wrong_mask.sum())


def evaluate(cfg_path):
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    device   = "cuda" if torch.cuda.is_available() else "cpu"
    save_dir = Path(cfg["paths"]["outputs"])
    save_dir.mkdir(parents=True, exist_ok=True)

    _, _, test_loader, _, _, _ = build_loaders(cfg)

    model = build_model(cfg, device)
    load_checkpoint(model, cfg["paths"]["best_model"], device)
    model.eval()

    use_tta = cfg["evaluation"]["tta"]
    print(f"Running evaluation (TTA={'on' if use_tta else 'off'})...")

    if use_tta:
        labels, probs, paths = collect_predictions_tta(model, test_loader, device, cfg)
    else:
        labels, probs, paths = collect_predictions(model, test_loader, device)

    threshold, _ = find_best_threshold(labels, probs)
    preds        = (probs >= threshold).astype(int)

    auc  = roc_auc_score(labels, probs)
    ap   = average_precision_score(labels, probs)
    rec  = recall_score(labels, preds)
    prec = precision_score(labels, preds)
    f1   = f1_score(labels, preds)
    bs   = brier_score_loss(labels, probs)

    print(f"\nTest results")
    print(f"  AUC       : {auc:.4f}")
    print(f"  AP        : {ap:.4f}")
    print(f"  Recall    : {rec:.4f}")
    print(f"  Precision : {prec:.4f}")
    print(f"  F1        : {f1:.4f}")
    print(f"  Brier     : {bs:.4f}")
    print(f"  Threshold : {threshold:.2f}")
    print(f"\n{classification_report(labels, preds, target_names=CLASSES)}")

    plot_roc_pr(labels, probs, save_dir)
    plot_confusion(labels, preds, threshold, save_dir)
    plot_score_distribution(labels, probs, threshold, save_dir)

    n_errors = error_analysis(model, labels, probs, paths, threshold, cfg, device, save_dir)
    print(f"\nError analysis: {n_errors} misclassified out of {len(labels)}")

    metrics = {
        "auc":               round(auc,  4),
        "average_precision": round(ap,   4),
        "recall":            round(rec,  4),
        "precision":         round(prec, 4),
        "f1":                round(f1,   4),
        "brier_score":       round(bs,   4),
        "threshold":         round(threshold, 2),
        "n_errors":          n_errors,
        "n_test":            len(labels),
    }

    with open(save_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nAll outputs saved to {save_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    evaluate(args.config)