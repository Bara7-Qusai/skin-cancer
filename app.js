# -*- coding: utf-8 -*-
"""
DermAI — DenseNet121 Skin Lesion Classifier
Features: Upload / Drag-Drop / Camera Capture / Grad-CAM Visualization
Run:  python app.py
Prod: gunicorn -w 2 -b 0.0.0.0:5000 app:app
"""

from flask import Flask, request, jsonify, render_template
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import io
import os
import base64
import numpy as np

# ─── Config ──────────────────────────────────────────────────────────────────
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "DenseNet121_isic_best.pth"
CLASSES    = ["Benign", "Malignant"]
IMG_SIZE   = 224

# ─── Transforms ──────────────────────────────────────────────────────────────
val_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
STD  = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)

# ─── Model ───────────────────────────────────────────────────────────────────
def load_model():
    m = models.densenet121(weights=None)
    num_features = m.classifier.in_features
    m.classifier = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(num_features, 2)
    )
    if os.path.exists(MODEL_PATH):
        m.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        print(f"✅  Loaded weights from {MODEL_PATH}")
    else:
        print(f"⚠️   {MODEL_PATH} not found — demo mode (random weights)")
    m = m.to(DEVICE)
    m.eval()
    return m

model = load_model()

# ─── Grad-CAM (pure PyTorch, no extra lib needed) ────────────────────────────
class GradCAM:
    def __init__(self, model, target_layer):
        self.model        = model
        self.target_layer = target_layer
        self.gradients    = None
        self.activations  = None
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate(self, input_tensor, class_idx=None):
        self.model.zero_grad()
        output = self.model(input_tensor)

        if class_idx is None:
            class_idx = output.argmax(dim=1).item()

        score = output[0, class_idx]
        score.backward()

        # Pool gradients across channels
        pooled_grads = self.gradients.mean(dim=[0, 2, 3])  # (C,)
        cam = self.activations[0]                            # (C, H, W)

        for i in range(cam.shape[0]):
            cam[i] *= pooled_grads[i]

        heatmap = cam.mean(dim=0).cpu().numpy()
        heatmap = np.maximum(heatmap, 0)

        if heatmap.max() > 0:
            heatmap /= heatmap.max()

        return heatmap, output

# Attach GradCAM to denseblock4
gradcam = GradCAM(model, model.features.denseblock4)

# ─── Colormap: jet ───────────────────────────────────────────────────────────
def apply_jet_colormap(heatmap_norm):
    """Convert a 0-1 float heatmap to an RGB jet colormap image (numpy H×W×3 uint8)."""
    h = np.clip(heatmap_norm, 0, 1)
    r = np.clip(1.5 - abs(4*h - 3), 0, 1)
    g = np.clip(1.5 - abs(4*h - 2), 0, 1)
    b = np.clip(1.5 - abs(4*h - 1), 0, 1)
    rgb = np.stack([r, g, b], axis=-1)
    return (rgb * 255).astype(np.uint8)

def overlay_gradcam(img_pil, heatmap, alpha=0.45):
    """Overlay Grad-CAM heatmap on original PIL image."""
    import cv2
    w, h = img_pil.size

    # Resize heatmap to original image size
    hm_resized = Image.fromarray((heatmap * 255).astype(np.uint8)).resize((w, h), Image.BILINEAR)
    hm_arr     = np.array(hm_resized) / 255.0

    colored = apply_jet_colormap(hm_arr)
    colored_pil = Image.fromarray(colored)

    orig_arr    = np.array(img_pil).astype(np.float32)
    colored_arr = colored_pil.convert("RGB")
    colored_arr = np.array(colored_arr).astype(np.float32)

    blended = (1 - alpha) * orig_arr + alpha * colored_arr
    blended = np.clip(blended, 0, 255).astype(np.uint8)
    return Image.fromarray(blended)

def pil_to_base64(img_pil, quality=88):
    buf = io.BytesIO()
    img_pil.save(buf, format="JPEG", quality=quality)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()

# ─── Inference + Grad-CAM ────────────────────────────────────────────────────
def predict_with_gradcam(image_bytes: bytes):
    img_orig = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img_orig.thumbnail((600, 600))

    img_t = val_transform(img_orig).unsqueeze(0).to(DEVICE)
    img_t.requires_grad_(True)

    heatmap, outputs = gradcam.generate(img_t)

    probs         = torch.softmax(outputs, dim=1)[0]
    pred_idx      = int(probs.argmax())
    pred_class    = CLASSES[pred_idx]
    benign_prob   = float(probs[0])
    malignant_prob= float(probs[1])
    confidence    = float(probs[pred_idx])

    # Build overlay
    display_img  = img_orig.copy()
    overlay      = overlay_gradcam(display_img, heatmap, alpha=0.45)

    # Heatmap only (jet-colored)
    hm_resized   = Image.fromarray(
        (heatmap * 255).astype(np.uint8)
    ).resize(img_orig.size, Image.BILINEAR)
    hm_colored   = Image.fromarray(
        apply_jet_colormap(np.array(hm_resized) / 255.0)
    )

    return {
        "prediction":       pred_class,
        "confidence":       round(confidence * 100, 2),
        "benign_prob":      round(benign_prob * 100, 2),
        "malignant_prob":   round(malignant_prob * 100, 2),
        "preview":          pil_to_base64(img_orig),
        "gradcam_overlay":  pil_to_base64(overlay),
        "gradcam_heatmap":  pil_to_base64(hm_colored),
    }

# ─── Flask App ───────────────────────────────────────────────────────────────
app = Flask(__name__, template_folder="templates", static_folder="static")
app.config["MAX_CONTENT_LENGTH"] = 20 * 1024 * 1024   # 20 MB

ALLOWED_TYPES = {"image/jpeg", "image/jpg", "image/png", "image/webp"}

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    # Accept either file upload OR base64 from camera
    if "file" in request.files:
        file = request.files["file"]
        if file.content_type not in ALLOWED_TYPES:
            return jsonify({"error": "Unsupported file type. Use JPEG or PNG."}), 400
        contents = file.read()

    elif request.is_json and "image_b64" in request.json:
        # Camera capture path
        b64_data = request.json["image_b64"]
        if b64_data.startswith("data:"):
            b64_data = b64_data.split(",", 1)[1]
        try:
            contents = base64.b64decode(b64_data)
        except Exception:
            return jsonify({"error": "Invalid base64 image data."}), 400
    else:
        return jsonify({"error": "No image provided."}), 400

    try:
        result = predict_with_gradcam(contents)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500

@app.route("/health")
def health():
    return jsonify({
        "status": "ok",
        "device": str(DEVICE),
        "model":  MODEL_PATH,
        "gradcam": "enabled"
    })

if __name__ == "__main__":
    print(f"🚀  DermAI running on http://0.0.0.0:5000  |  device: {DEVICE}")
    app.run(host="0.0.0.0", port=5000, debug=False)
