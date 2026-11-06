# DeepGuard AI — Deepfake Detection + Grad-CAM++

Full-stack starter project:
- Google Colab notebook: `DeepGuard_AI_Colab.ipynb`
- VS Code application: Flask API + React/Vite frontend
- Model: pretrained Vision Transformer (ViT), fine-tuned for `Fake=0`, `Real=1`
- XAI: Grad-CAM++ over the final ViT transformer block

## Important dataset notes
The Kaggle DFD distribution may contain videos, extracted frames, or nested folders. Do not assume folder names. Inspect the extracted tree first, then set `REAL_FOLDER_TOKENS` and `FAKE_FOLDER_TOKENS` in the notebook. Any ambiguous/unmatched video is deliberately skipped and reported.

The notebook samples frames and extracts faces. It keeps `source_video` in metadata and performs group-aware splitting by source video to reduce frame leakage. If the dataset includes multiple videos of the same identity or source recording, consider grouping by the strongest shared source/identity identifier as well.

## 1. Google Colab
1. Upload `DeepGuard_AI_Colab.ipynb` to Google Colab.
2. Runtime → Change runtime type → GPU.
3. Run cells in order.
4. Generate a Kaggle API token from Kaggle Settings → API, upload `kaggle.json` when prompted.
5. Inspect the dataset tree and configure label tokens before extraction.
6. At the end, download `best_model.pth` and `class_names.json`.

The notebook has configurable limits for quick smoke tests. Increase `MAX_FACES_PER_VIDEO` and/or reduce `FRAME_INTERVAL_SECONDS` for a fuller run after confirming the pipeline works.

## 2. VS Code
Requirements: Python 3.10+ and Node.js 20+.

### Backend
```bash
cd backend
python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# macOS/Linux:
source .venv/bin/activate
pip install -r requirements.txt
```
Copy the exported `best_model.pth` into `models/best_model.pth`.
Copy `class_names.json` into `models/class_names.json`.
Set `MODEL_PATH=../models/best_model.pth` if running from `backend/`, or use the default path.
```bash
python app.py
```
API: `http://127.0.0.1:5000`

### Frontend
In another terminal:
```bash
cd frontend
npm install
npm run dev
```
Open the Vite URL shown in terminal (normally `http://localhost:5173`).

## API
`POST /api/predict` — multipart form field `image`.
Returns prediction, confidence, class probabilities, Grad-CAM++ heatmap and overlay as base64 PNG data URLs.
`GET /api/health` — readiness check.

## Evaluation
The notebook reports accuracy, precision, recall, F1, ROC-AUC (when possible), classification report and confusion matrix. Do not report metrics until the held-out test set has actually been evaluated.

## Limitations
- A single-frame image classifier does not use temporal evidence from the original video.
- Grad-CAM++ highlights regions that influenced this model; it is not proof of manipulation.
- Confidence is model confidence, not a calibrated probability of authenticity.
- Test on identities, compression levels, cameras and manipulation methods not seen during training.
- This is a screening aid, not a forensic certification tool.
