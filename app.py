import sys
import json
import warnings
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from PIL import Image

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).parent / "src"))

from dataset import CLASSES, get_transforms, get_tta_transforms
from model import build_model, load_checkpoint
from gradcam import GradCAM, render_overlay

BASE         = Path(__file__).parent
METRICS_PATH = BASE / "outputs" / "metrics.json"
HISTORY_PATH = BASE / "outputs" / "history.json"
BEST_MODEL   = BASE / "checkpoints" / "best.pt"
DATA_ROOT    = BASE / "data" / "chest_xray"
IMG_SIZE     = 224

st.set_page_config(page_title="CXR-Detect", layout="wide")


@st.cache_resource
def load_pipeline():
    import yaml
    import torch

    with open(BASE / "config.yaml") as f:
        cfg = yaml.safe_load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model  = build_model(cfg, device)
    load_checkpoint(model, str(BEST_MODEL), device)
    model.eval()
    return model, cfg, device


def model_ready():
    return BEST_MODEL.exists()


def get_threshold():
    if METRICS_PATH.exists():
        return json.load(open(METRICS_PATH))["threshold"]
    return 0.5


def predict_image(img_pil, model, device, cfg, use_tta):
    import torch
    img_size = cfg["data"]["img_size"]

    if use_tta:
        tfs   = get_tta_transforms(img_size)
        probs = []
        for tf in tfs:
            t = tf(img_pil).unsqueeze(0).to(device)
            with torch.no_grad():
                prob = torch.softmax(model(t), dim=1)[0, 1].item()
            probs.append(prob)
        return float(np.mean(probs))

    tf = get_transforms(img_size, "val", cfg)
    t  = tf(img_pil).unsqueeze(0).to(device)
    with torch.no_grad():
        return torch.softmax(model(t), dim=1)[0, 1].item()


def run_gradcam(img_pil, model, cfg):
    img_size    = cfg["data"]["img_size"]
    tf          = get_transforms(img_size, "val", cfg)
    img_t       = tf(img_pil)
    gc          = GradCAM(model, model.get_gradcam_layer())
    cam, _, _   = gc.generate(img_t)
    img_np, _, overlay = render_overlay(img_pil, cam, img_size)
    gc.remove_hooks()
    return cam, overlay, img_np


def load_test_samples(n=3):
    samples = []
    for cls in CLASSES:
        folder = DATA_ROOT / "test" / cls
        if folder.exists():
            for f in sorted(folder.iterdir())[:n]:
                samples.append((f, cls))
    return samples


st.sidebar.title("CXR-Detect")
st.sidebar.markdown("Pneumonia detection from chest X-rays")
st.sidebar.markdown("---")

if METRICS_PATH.exists():
    m = json.load(open(METRICS_PATH))
    st.sidebar.markdown(f"AUC &nbsp;&nbsp;&nbsp; `{m['auc']}`")
    st.sidebar.markdown(f"Recall &nbsp; `{m['recall']}`")
    st.sidebar.markdown(f"F1 &nbsp;&nbsp;&nbsp;&nbsp; `{m['f1']}`")

st.sidebar.markdown("---")
page = st.sidebar.radio("Navigation", ["Classify", "Grad-CAM", "Training History", "Performance"], label_visibility="collapsed")

if not model_ready():
    st.warning("No trained model found. Run `python src/train.py` first.")
    st.stop()

model, cfg, device = load_pipeline()
threshold          = get_threshold()


if page == "Classify":
    st.title("Classify Chest X-Ray")

    col1, col2 = st.columns([1, 2])

    with col1:
        uploaded = st.file_uploader("Upload X-ray", type=["jpg", "jpeg", "png"])
        use_tta  = st.checkbox("Test-time augmentation", value=True)

        samples = load_test_samples(n=3)
        labels  = [f"{cls} — {p.name}" for p, cls in samples]
        chosen  = st.selectbox("Or use a test sample", ["— select —"] + labels)

    img_pil    = None
    true_label = None

    if uploaded:
        img_pil = Image.open(uploaded).convert("RGB")
    elif chosen != "— select —":
        idx, true_label = labels.index(chosen), samples[labels.index(chosen)][1]
        img_pil = Image.open(samples[idx][0]).convert("RGB")

    with col2:
        if img_pil:
            prob = predict_image(img_pil, model, device, cfg, use_tta)
            pred = CLASSES[int(prob >= threshold)]

            c1, c2, c3 = st.columns(3)
            c1.metric("Pneumonia probability", f"{prob:.1%}")
            c2.metric("Prediction", pred)
            if true_label:
                correct = pred == true_label
                c3.metric("Ground truth", true_label,
                          delta="Correct" if correct else "Wrong",
                          delta_color="normal" if correct else "inverse")

            cam, overlay, img_np = run_gradcam(img_pil, model, cfg)

            fig, axes = plt.subplots(1, 3, figsize=(13, 4))
            axes[0].imshow(img_np);          axes[0].set_title("Input");    axes[0].axis("off")
            axes[1].imshow(cam, cmap="jet"); axes[1].set_title("Grad-CAM"); axes[1].axis("off")
            axes[2].imshow(overlay);         axes[2].set_title("Overlay");  axes[2].axis("off")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
        else:
            st.info("Upload an X-ray or select a test sample.")


elif page == "Grad-CAM":
    st.title("Grad-CAM Visualisations")
    st.markdown("Which lung regions does the model focus on?")

    samples = load_test_samples(n=2)
    if not samples:
        st.warning("No test images found.")
        st.stop()

    fig, axes = plt.subplots(len(samples), 3, figsize=(13, 4 * len(samples)))
    if len(samples) == 1:
        axes = axes[np.newaxis, :]

    for row, (path, true_label) in enumerate(samples):
        img_pil = Image.open(path).convert("RGB")
        prob    = predict_image(img_pil, model, device, cfg, use_tta=False)
        pred    = CLASSES[int(prob >= threshold)]
        cam, overlay, img_np = run_gradcam(img_pil, model, cfg)

        axes[row, 0].imshow(img_np);          axes[row, 0].set_title(f"GT: {true_label}"); axes[row, 0].axis("off")
        axes[row, 1].imshow(cam, cmap="jet"); axes[row, 1].set_title("Grad-CAM");          axes[row, 1].axis("off")
        axes[row, 2].imshow(overlay);         axes[row, 2].set_title(f"Pred: {pred} ({prob:.1%})"); axes[row, 2].axis("off")

    plt.tight_layout()
    st.pyplot(fig)
    plt.close()


elif page == "Training History":
    st.title("Training History")

    if not HISTORY_PATH.exists():
        st.info("No training history found. Run `python src/train.py` first.")
        st.stop()

    history = json.load(open(HISTORY_PATH))
    epochs  = list(range(1, len(history["train_loss"]) + 1))

    fig, axes = plt.subplots(1, 2, figsize=(13, 4))

    axes[0].plot(epochs, history["train_loss"], label="Train", lw=2)
    axes[0].plot(epochs, history["val_loss"],   label="Val",   lw=2)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Loss")
    axes[0].legend()

    axes[1].plot(epochs, history["train_auc"], label="Train", lw=2)
    axes[1].plot(epochs, history["val_auc"],   label="Val",   lw=2)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("AUC")
    axes[1].set_title("AUC-ROC")
    axes[1].legend()

    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    best_epoch = int(np.argmax(history["val_auc"])) + 1
    st.markdown(f"Best val AUC: `{max(history['val_auc']):.4f}` at epoch `{best_epoch}`")


elif page == "Performance":
    st.title("Test Set Performance")

    if not METRICS_PATH.exists():
        st.info("Run `python src/evaluate.py` to generate metrics.")
        st.stop()

    m = json.load(open(METRICS_PATH))

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("AUC",       m["auc"])
    c2.metric("Recall",    m["recall"])
    c3.metric("Precision", m["precision"])
    c4.metric("F1",        m["f1"])
    c5.metric("Brier",     m["brier_score"])

    st.markdown("")

    for title, fname in [
        ("ROC & PR Curves",       "roc_pr.png"),
        ("Confusion Matrix",      "confusion_matrix.png"),
        ("Score Distribution",    "score_distribution.png"),
        ("Error Analysis",        "error_gradcam.png"),
    ]:
        path = BASE / "outputs" / fname
        if path.exists():
            st.markdown(f"**{title}**")
            st.image(str(path), use_column_width=True)
        else:
            st.markdown(f"*{title} not found — run evaluate.py*")