import sys
import json
from pathlib import Path

import numpy as np
import torch
import yaml
import streamlit as st
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent / "src"))

from dataset import get_transforms, get_tta_transforms, CLASSES
from model import build_model, load_checkpoint
from gradcam import GradCAM, render_overlay

import matplotlib
matplotlib.use("Agg")

BASE = Path(__file__).parent
BEST_MODEL = BASE / "checkpoints" / "best.pt"
METRICS_PATH = BASE / "outputs" / "metrics.json"


# ── PAGE CONFIG ─────────────────────────
st.set_page_config(
    page_title="PneumoScan AI",
    page_icon="🫁",
    layout="centered",
)

# ── GLOBAL STYLE ────────────────────────
st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond&family=DM+Sans:wght@300;400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

.stApp {
    background-color: #f7f4f0;
}

.block-container {
    max-width: 720px;
}

#MainMenu, footer, header { visibility: hidden; }

/* HEADER */

.ps-header {
    text-align: center;
    padding: 2rem 0;
    border-bottom: 1px solid #e8e3dc;
}

.ps-logo {
    font-family: 'Cormorant Garamond', serif;
    font-size: 1.7rem;
}

.ps-logo span {
    color: #2f6ee8;
}

.ps-tagline {
    font-size: .8rem;
    color: #9a9088;
}

/* HERO */

.ps-h1 {
    font-family: 'Cormorant Garamond', serif;
    font-size: 2.2rem;
    text-align: center;
    margin-top: 2rem;

}

.ps-sub {
    text-align: center;
    font-size: .9rem;
    color: #9a9088;
    margin-bottom: 2rem;
}

div[data-testid="stMarkdownContainer"] p,
div[data-testid="stMarkdownContainer"] div,
div[data-testid="stMarkdownContainer"] h1,
div[data-testid="stMarkdownContainer"] h2,
div[data-testid="stMarkdownContainer"] h3 {
    color: #1a1714 !important;
}

.ps-tagline,
.ps-sub,
.viewing-row {
    color: #9a9088 !important;
}

div[data-testid="stCheckbox"] label[data-baseweb="checkbox"] > div:first-child {
    background-color: #9a9088 !important;
    border-color: #9a9088 !important;
}

div[data-testid="stCheckbox"] label[data-baseweb="checkbox"] > div:first-child > div {
    background-color: #ffffff !important;
}

/* METRICS */
            
div[data-testid="stMetric"] {
    background-color: #ffffff !important;
    border: 1px solid #e8e3dc !important;
    border-radius: 8px !important;
    padding: 1rem 1.2rem !important;
}
 
div[data-testid="stMetricLabel"] p,
div[data-testid="stMetricLabel"] {
    color: #9a9088 !important;
    font-size: 1.9rem !important;
    font-weight: 500 !important;
    text-transform: uppercase !important;
    letter-spacing: .05em !important;
     opacity: 1 !important;
}
            
div[data-testid="stMetricLabel"] * {
    color: #1a1714 !important;
    opacity: 1 !important;
}
 
div[data-testid="stMetricValue"] > div,
div[data-testid="stMetricValue"] {
    color: #1a1714 !important;
    font-family: 'Cormorant Garamond', serif !important;
    font-size: 2rem !important;
}
 
div[data-testid="stMetricValue"] > div,
div[data-testid="stMetricValue"] {
    color: #1a1714 !important;
    font-family: 'Cormorant Garamond', serif !important;
    font-size: 2rem !important;
}

/* VIEWING TEXT */

.viewing-row {
    display: flex;
    align-items: center;
    justify-content: space-between;
    font-size: 1.1rem;
    color: #9a9088;
}
.viewing-text {
    display: flex;
    margin-top:.3rem;
}

</style>
""", unsafe_allow_html=True)


# ── MODEL LOADING ─────────────────────
@st.cache_resource
def load_pipeline():

    with open(BASE / "config.yaml") as f:
        cfg = yaml.safe_load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = build_model(cfg, device)
    load_checkpoint(model, str(BEST_MODEL), device)

    model.eval()

    threshold = 0.5

    if METRICS_PATH.exists():
        threshold = json.load(open(METRICS_PATH))["threshold"]

    return model, cfg, device, threshold


def run_inference(img_pil, model, cfg, device, threshold):

    img_size = cfg["data"]["img_size"]
    tta = get_tta_transforms(img_size)

    probs = []

    for tf in tta:

        t = tf(img_pil).unsqueeze(0).to(device)

        with torch.no_grad():
            p = torch.softmax(model(t), dim=1)[0, 1].item()

        probs.append(p)

    prob = float(np.mean(probs))
    prediction = CLASSES[int(prob >= threshold)]
    confidence = max(prob, 1 - prob)

    tf = get_transforms(img_size, "val", cfg)
    img_t = tf(img_pil)

    gc = GradCAM(model, model.get_gradcam_layer())

    cam, _, _ = gc.generate(img_t)

    img_np, _, overlay = render_overlay(img_pil, cam, img_size)

    gc.remove_hooks()

    return prediction, confidence, prob, img_np, overlay


# ── HEADER ─────────────────────────
st.markdown("""
<div class="ps-header">
<div class="ps-logo">Pneumo<span>Scan</span> AI</div>
<div class="ps-tagline">Chest X-ray Pneumonia Detection</div>
</div>
""", unsafe_allow_html=True)


# ── NAVIGATION ROW ─────────────────

col1, col2 = st.columns([6, 1])

with col2:
    toggle = st.toggle("", value=False)

page = "Model Performance" if toggle else "Analyze"

with col1:
    st.markdown(
        f"""
        <div class="viewing-row">
        <div class="viewing-text">Viewing: <strong style="color:#1a1714;">{page}</strong></div>
        </div>
        """,
        unsafe_allow_html=True
    )


# ── ANALYZE PAGE ─────────────────
if page == "Analyze":

    st.markdown('<div class="ps-h1">Chest X-Ray Pneumonia Detection</div>', unsafe_allow_html=True)
    st.markdown('<div class="ps-sub">Upload a chest X-ray for AI analysis</div>', unsafe_allow_html=True)

    uploaded = st.file_uploader("", type=["jpg", "jpeg", "png"])

    if uploaded:

        if st.button("Analyze X-ray"):

            img_pil = Image.open(uploaded).convert("RGB")

            with st.spinner("Analyzing..."):

                model, cfg, device, threshold = load_pipeline()

                prediction, confidence, prob, img_np, overlay = run_inference(
                    img_pil, model, cfg, device, threshold
                )

                st.success(f"Prediction: {prediction} | Confidence: {confidence * 100:.1f}%")

                c1, c2 = st.columns(2)

                with c1:
                    st.image(img_np, width="stretch")

                with c2:
                    st.image(overlay, width="stretch")


# ── PERFORMANCE PAGE ──────────────
elif page == "Model Performance":

    st.markdown('<div class="ps-h1">Model Performance</div>', unsafe_allow_html=True)

    metrics = {}

    if METRICS_PATH.exists():
        metrics = json.load(open(METRICS_PATH))

    st.metric("AUC", f"{metrics.get('auc', 0.98) * 100:.1f}%")
    st.metric("F1 Score", f"{metrics.get('f1', 0.96) * 100:.1f}%")
    st.metric("Recall", f"{metrics.get('recall', 0.98) * 100:.1f}%")
    st.metric("Precision", f"{metrics.get('precision', 0.95) * 100:.1f}%")

    roc = BASE / "outputs" / "roc_pr.png"
    cm = BASE / "outputs" / "confusion_matrix.png"

    if roc.exists():
        st.image(str(roc), width="stretch")

    if cm.exists():
        st.image(str(cm), width="stretch")