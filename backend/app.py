import os
import io
import base64
from pathlib import Path
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import numpy as np
import cv2

from model_utils import load_model, predict_with_explanation

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = Path(os.getenv("MODEL_PATH", ROOT / "models" / "best_model.pth"))
MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "10"))

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_MB * 1024 * 1024
CORS(app, resources={r"/api/*": {"origins": os.getenv("FRONTEND_ORIGIN", "http://localhost:5173")}})

model = None
device = None

def png_data_url(array):
    if array.ndim == 2:
        image = Image.fromarray(array.astype(np.uint8), mode="L").convert("RGB")
    else:
        image = Image.fromarray(array.astype(np.uint8), mode="RGB")
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return "data:image/png;base64," + encoded

@app.get("/api/health")
def health():
    return jsonify({
        "status": "ok",
        "model_loaded": model is not None,
        "device": str(device) if device is not None else None
    })

@app.post("/api/predict")
def predict():
    if model is None:
        return jsonify({"error": "Model is not loaded. Add models/best_model.pth and restart the API."}), 503
    if "image" not in request.files:
        return jsonify({"error": "Upload an image using multipart field 'image'."}), 400

    upload = request.files["image"]
    if not upload.filename:
        return jsonify({"error": "No image selected."}), 400

    try:
        image = Image.open(upload.stream).convert("RGB")
        if image.width < 64 or image.height < 64:
            return jsonify({"error": "Image is too small. Upload an image at least 64x64 pixels."}), 400

        result = predict_with_explanation(model, device, image)
        original = np.asarray(image.resize((224, 224)))
        return jsonify({
            "prediction": result["prediction"],
            "confidence": result["confidence"],
            "probabilities": result["probabilities"],
            "original": png_data_url(original),
            "heatmap": png_data_url(result["heatmap"]),
            "overlay": png_data_url(result["overlay"]),
            "explanation_note": "The heatmap shows regions influencing this model's prediction. It is not proof that the image is fake."
        })
    except Exception as exc:
        app.logger.exception("Prediction failed")
        return jsonify({"error": f"Could not analyze image: {str(exc)}"}), 500

if __name__ == "__main__":
    if MODEL_PATH.exists():
        model, device = load_model(str(MODEL_PATH))
        print(f"Loaded model from {MODEL_PATH} on {device}")
    else:
        print(f"WARNING: model checkpoint not found: {MODEL_PATH}")
        print("The API will start, but /api/predict returns 503 until a checkpoint is added.")
    app.run(host="0.0.0.0", port=5000, debug=True)
