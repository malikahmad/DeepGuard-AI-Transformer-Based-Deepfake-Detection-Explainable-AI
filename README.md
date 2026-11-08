# DeepGuard AI — Transformer-Based Deepfake Detection & Explainable AI

DeepGuard AI is an image-classification prototype for distinguishing facial images labelled **Real** or **Fake**. It combines a Vision Transformer (ViT) workflow with Grad-CAM++ visualizations to make model outputs easier to inspect.

> **Research disclaimer:** This is an experimental image-analysis project, not a forensic verification tool. A prediction or heatmap is not proof that an image is authentic or manipulated.

## Highlights

- Image upload and Real/Fake classification.
- Vision Transformer model implemented with PyTorch.
- Prediction confidence display.
- Grad-CAM++ heatmaps and overlays for visual inspection.
- Model evaluation workflow for accuracy, precision, recall, F1-score, ROC-AUC, and confusion matrix.
- Flask inference backend and React + TypeScript frontend.

## Technology

| Layer | Technology |
|---|---|
| Language | Python |
| Deep learning | PyTorch |
| Architecture | Vision Transformer (ViT) |
| Explainability | Grad-CAM++ |
| Image processing | OpenCV, Pillow |
| Backend | Flask, REST API |
| Frontend | React, TypeScript, Vite |
| Experiment environment | Google Colab |

## How it works

1. The user uploads a facial image.
2. The backend validates and preprocesses the image.
3. The ViT model produces class scores.
4. The application displays the predicted class and confidence.
5. Grad-CAM++ produces a visualization of image regions associated with the model output.

Heatmaps are explanatory aids, not segmentation masks or evidence of model correctness.

## Repository layout

```text
DeepGuard-AI/
├── backend/
│   ├── app.py
│   ├── model_utils.py
│   └── requirements.txt
├── frontend/
│   ├── src/
│   ├── package.json
│   └── vite.config.ts
├── models/
│   ├── best_model.pth
│   └── class_names.json
├── notebooks/
│   └── DeepGuard_AI_Colab.ipynb
└── README.md
```

The layout above documents the intended project organization; filenames may differ if your local checkout has changed.

## Run locally

### 1. Clone

```bash
git clone https://github.com/malikahmad/DeepGuard-AI-Transformer-Based-Deepfake-Detection-Explainable-AI.git
cd DeepGuard-AI-Transformer-Based-Deepfake-Detection-Explainable-AI
```

### 2. Backend

Use a Python version compatible with the installed PyTorch dependencies.

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
pip install -r backend/requirements.txt
python backend/app.py
```

Place the trained checkpoint and class-name metadata in the location expected by the backend configuration.

### 3. Frontend

Open a second terminal:

```bash
cd frontend
npm install
npm run dev
```

Open the local URL printed by Vite.

## Evaluation checklist

Before reporting a headline score, evaluate on a held-out test set that was not used for training or model selection. Include:

- Dataset source, class balance, and split strategy.
- Accuracy, precision, recall, and F1-score.
- ROC-AUC where appropriate.
- Confusion matrix and per-class results.
- Cross-dataset or out-of-distribution testing where possible.
- Checks for duplicate images and data leakage.

**Measured results:** Add the actual run's metrics and evaluation date here. No performance score is asserted in this README.

## Limitations and next steps

- Current workflow is image-based; video-level temporal analysis is not claimed.
- Performance may change with compression, resizing, unseen sources, and manipulation methods.
- Confidence scores should not be interpreted as calibrated probabilities unless calibration has been tested.
- Grad-CAM++ may be unstable and does not establish causal reasoning.

Potential extensions include video-frame analysis, robustness testing, calibration, and independent cross-dataset evaluation.

## Author

**Ahmad Zulfiqar** — [GitHub](https://github.com/malikahmad) · [LinkedIn](https://www.linkedin.com/in/malik-ahmad-06abb31a1/)
