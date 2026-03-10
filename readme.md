# CXR-Detect

Chest X-ray pneumonia detection using a fine-tuned ResNet50. The model predicts whether a chest X-ray shows signs of pneumonia and highlights the lung regions that drove the prediction using Grad-CAM — making every decision auditable.



- Overview

Pneumonia kills over 2 million people annually, many of which go undiagnosed due to limited access to radiologists. CXR-Detect automates the screening step — flagging high-risk X-rays so clinicians can prioritise their attention where it matters most.

The model achieves ~97% AUC on the Kaggle chest X-ray dataset with a sensitivity of ~95% on pneumonia cases. Sensitivity is intentionally prioritised over precision — missing a real pneumonia case is more harmful than an unnecessary follow-up.



- Project Structure


CXR-Detect/
├── config.yaml               all hyperparameters in one place
├── prepare_data.py           downloads and structures the Kaggle dataset
├── app.py                    streamlit dashboard
├── requirements.txt
├── README.md
│
├── src/
│   ├── dataset.py            dataset class, augmentation pipeline, weighted sampler
│   ├── model.py              ResNet50 with custom head, freeze logic, ONNX export
│   ├── train.py              full training script — runs headless
│   ├── evaluate.py           evaluation, threshold tuning, error analysis, Grad-CAM
│   ├── predict.py            single image inference with optional Grad-CAM overlay
│   └── gradcam.py            Grad-CAM implementation using PyTorch hooks
│
├── notebooks/
│   └── exploration.ipynb     EDA only — class distribution, sample images, augmentation preview
│
├── data/
│   └── chest_xray/
│       ├── train/
│       ├── val/
│       └── test/
│
├── checkpoints/              model weights saved during training
└── outputs/                  metrics, plots, ONNX model




- Quickstart

1. Clone the repo

git clone https://github.com/ronak-2005/CXR-Detect.git
cd CXR-Detect


2. Install dependencies

pip install -r requirements.txt


3. Set up Kaggle API

# Download kaggle.json from kaggle.com → Settings → API → Create New Token
mkdir -p ~/.kaggle
mv ~/Downloads/kaggle.json ~/.kaggle/kaggle.json
chmod 600 ~/.kaggle/kaggle.json


4. Download and prepare data

python prepare_data.py


5. Train

python src/train.py --config config.yaml


6. Evaluate

python src/evaluate.py --config config.yaml


7. Launch the app

streamlit run app.py


8. Predict a single image

python src/predict.py path/to/xray.jpg --gradcam output.png




- Model

ResNet50 pretrained on ImageNet, fine-tuned in two phases:

- Phase 1 (epochs 1-4): backbone frozen, only the custom head trains
- Phase 2 (epoch 5+): layer3 and layer4 unfrozen, full end-to-end fine-tuning at a lower learning rate

This two-phase approach prevents catastrophic forgetting of the pretrained features while still adapting the deeper layers to the X-ray domain.

Head architecture:

Linear(2048 → 512) → BatchNorm → ReLU → Dropout(0.4)
→ Linear(512 → 128) → BatchNorm → ReLU → Dropout(0.2)
→ Linear(128 → 2)




- Training

Key decisions:

- Weighted random sampler — the dataset has a 3:1 pneumonia-to-normal imbalance. Rather than adjusting the loss function alone, the sampler ensures every batch sees balanced classes during training.
- Label smoothing (0.1) — prevents overconfidence. Early-stage pneumonia can look normal on an X-ray — hard labels would penalise the model for reasonable uncertainty.
- Warmup + cosine LR schedule — 2 epochs of linear warmup followed by cosine annealing to near-zero.
- Mixed precision (fp16) — roughly halves memory usage and speeds up training on CUDA.
- Early stopping (patience=8) — monitors validation AUC. Best checkpoint saved separately from the last checkpoint.
- Gradient clipping (max_norm=1.0) — stabilises training during the backbone unfreeze phase.



- Evaluation

Beyond standard metrics, `evaluate.py` runs:

- Threshold tuning — finds the threshold that maximises F1 on the val set before reporting test metrics
- Test-time augmentation (TTA) — averages predictions across 5 augmented versions of each test image
- Error analysis — finds the most confidently wrong predictions, generates Grad-CAM for them specifically
- Calibration — Brier score to verify predicted probabilities are trustworthy



- Grad-CAM

Implemented from scratch using PyTorch forward and backward hooks — not a library wrapper. Gradients are backpropagated to `layer4[-1]` of ResNet50, weighted by their spatial average, and upsampled to input resolution.

This is critical for clinical trust. A model can achieve high accuracy for the wrong reasons — by picking up scanner labels, patient positioning, or equipment artefacts. Grad-CAM makes it easy to catch this during development.



- Results

| Metric | Value |

Test AUC ~0.97 
Sensitivity (pneumonia recall) ~95% 
Specificity (normal recall) ~88% 
F1 ~0.94 



- Dataset

Kaggle Chest X-Ray Images (Pneumonia)
https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia

5,863 JPEG images across train, val, and test splits. The original Kaggle val split contains only 16 images — `prepare_data.py` discards it and carves a proper 15% validation set from train instead.



- Requirements

- Python 3.10+
- PyTorch 2.1+
- CUDA-capable GPU recommended (CPU works but training will be slow)



- Stack

Python · PyTorch · torchvision · scikit-learn · Streamlit · Pillow · matplotlib · seaborn · PyYAML