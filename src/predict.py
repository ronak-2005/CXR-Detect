import argparse
import json
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image

from dataset import get_transforms, get_tta_transforms, CLASSES
from model import build_model, load_checkpoint
from gradcam import GradCAM, render_overlay, plot_gradcam


def load_pipeline(cfg_path):
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model  = build_model(cfg, device)
    load_checkpoint(model, cfg["paths"]["best_model"], device)
    model.eval()

    metrics_path = Path(cfg["paths"]["metrics"])
    if metrics_path.exists():
        with open(metrics_path) as f:
            metrics = json.load(f)
        threshold = metrics["threshold"]
    else:
        threshold = cfg["evaluation"]["threshold"]

    return model, cfg, device, threshold


def predict(img_path, model, cfg, device, threshold, use_tta=True, save_gradcam=None):
    img_size = cfg["data"]["img_size"]
    img_pil  = Image.open(img_path).convert("RGB")

    if use_tta:
        tta_tfs   = get_tta_transforms(img_size)
        probs_tta = []
        for tf in tta_tfs:
            t = tf(img_pil).unsqueeze(0).to(device)
            with torch.no_grad():
                prob = torch.softmax(model(t), dim=1)[0, 1].item()
            probs_tta.append(prob)
        prob = float(np.mean(probs_tta))
    else:
        tf   = get_transforms(img_size, "val", cfg)
        t    = tf(img_pil).unsqueeze(0).to(device)
        with torch.no_grad():
            prob = torch.softmax(model(t), dim=1)[0, 1].item()

    prediction = CLASSES[int(prob >= threshold)]
    confidence = max(prob, 1 - prob)

    if save_gradcam:
        tf      = get_transforms(img_size, "val", cfg)
        img_t   = tf(img_pil)
        gc      = GradCAM(model, model.get_gradcam_layer())
        cam, _, _ = gc.generate(img_t)
        img_np, _, overlay = render_overlay(img_pil, cam, img_size)
        title   = f"{prediction} ({prob:.1%})"
        fig     = plot_gradcam(img_np, cam, overlay, title=title, save_path=save_gradcam)
        gc.remove_hooks()
        import matplotlib.pyplot as plt
        plt.close(fig)

    return {
        "path":             str(img_path),
        "prediction":       prediction,
        "pneumonia_prob":   round(prob, 4),
        "normal_prob":      round(1 - prob, 4),
        "confidence":       round(confidence, 4),
        "threshold":        round(threshold, 2),
        "tta":              use_tta,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("image",       help="path to chest x-ray image")
    parser.add_argument("--config",    default="config.yaml")
    parser.add_argument("--no-tta",    action="store_true")
    parser.add_argument("--gradcam",   default=None, help="save gradcam to this path")
    args = parser.parse_args()

    model, cfg, device, threshold = load_pipeline(args.config)

    result = predict(
        args.image,
        model, cfg, device, threshold,
        use_tta=not args.no_tta,
        save_gradcam=args.gradcam
    )

    print(json.dumps(result, indent=2))