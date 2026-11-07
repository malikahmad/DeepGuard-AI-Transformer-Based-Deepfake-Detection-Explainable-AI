# 🛡️ DeepGuard AI – Transformer-Based Deepfake Detection & Explainable AI

**DeepGuard AI** is an AI-powered deepfake detection system that uses Transformer-based deep learning to classify facial images as **Real or Fake**. It combines Vision Transformer (ViT) architecture with Explainable AI techniques to make model predictions more interpretable.

The system provides image-based predictions, confidence scores, and Grad-CAM++ heatmap visualizations highlighting regions that influenced the model's decision.

## 🚀 Key Features

- 🔍 **Deepfake Detection:** Classifies facial images as Real or Fake.
- 🧠 **Vision Transformer (ViT):** Transformer-based image classification.
- 📊 **Model Evaluation:** Accuracy, Precision, Recall, F1-score, ROC-AUC, and confusion matrix.
- 🔥 **Explainable AI:** Grad-CAM++ heatmaps and image overlays.
- 🖼️ **Image Upload:** Upload an image for deepfake analysis.
- ⚡ **Flask REST API:** Backend model inference and processing.
- 💻 **Modern Dashboard:** Responsive React and TypeScript frontend.
- 📈 **Model Insights:** Displays prediction confidence and visual explanations.

## 🛠️ Tech Stack

| Category | Technologies |
|---|---|
| Programming | Python |
| Deep Learning | PyTorch |
| Model Architecture | Vision Transformer (ViT) |
| Explainable AI | Grad-CAM++ |
| Computer Vision | OpenCV, PIL |
| Backend | Flask, REST API |
| Frontend | React, TypeScript, Vite |
| Model Training | Google Colab |
| Dataset | Deep Fake Detection Dataset (DFD) |

## 🏗️ System Architecture

1. User uploads a facial image.
2. The backend validates and preprocesses the image.
3. The trained Vision Transformer analyzes the image.
4. The model predicts Real or Fake.
5. Prediction confidence scores are generated.
6. Grad-CAM++ creates a heatmap visualization.
7. The frontend displays the prediction, original image, heatmap, and overlay.

## 📂 Project Structure

```text
DeepGuard-AI/
│
├── backend/
│   ├── app.py
│   ├── model_utils.py
│   └── requirements.txt
│
├── frontend/
│   ├── src/
│   │   ├── App.tsx
│   │   └── styles.css
│   ├── package.json
│   └── vite.config.ts
│
├── models/
│   ├── best_model.pth
│   └── class_names.json
│
├── notebooks/
│   └── DeepGuard_AI_Colab.ipynb
│
├── README.md
└── .gitignore
```

## 📊 Model Evaluation

The model is evaluated using:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- Classification Report
- Confusion Matrix

Actual evaluation results should be added here after training and testing.

## 🔥 Explainable AI – Grad-CAM++

Grad-CAM++ generates visual heatmaps highlighting image regions that contribute to the model's prediction.

These visualizations help interpret model behavior but should not be treated as definitive proof of image manipulation.

## ⚙️ Installation and Setup

### 1. Clone the repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd DeepGuard-AI
```

### 2. Backend setup

```bash
python -m venv .venv
```

Activate the environment on Windows:

```powershell
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r backend/requirements.txt
```

Place your trained model files inside the `models/` directory.

### 3. Start Flask backend

```bash
python backend/app.py
```

### 4. Frontend setup

Open another terminal:

```bash
cd frontend
npm install
npm run dev
```

Open the local URL displayed by Vite in your browser.

## 🎯 Project Objectives

- Explore Transformer-based deepfake detection.
- Apply deep learning to digital image authenticity analysis.
- Integrate Explainable AI into image classification.
- Build a complete AI-powered web application.
- Evaluate model performance using standard classification metrics.

## 🔮 Future Improvements

- Video-based deepfake detection.
- Face tracking and multi-frame analysis.
- Real-time video inference.
- Model robustness testing against compression and image manipulation.
- Model calibration and improved generalization.
- Expanded datasets and cross-dataset evaluation.

## 👨‍💻 Author

**Ahmad Zulfiqar**

AI/ML Engineer | Machine Learning | Deep Learning | Computer Vision | Generative AI

## ⚠️ Disclaimer

DeepGuard AI is an experimental research and development project. Its predictions are not definitive evidence of authenticity or manipulation. Performance depends on training data, model quality, and evaluation conditions.

---

⭐ If you find this project interesting, consider giving the repository a star!
