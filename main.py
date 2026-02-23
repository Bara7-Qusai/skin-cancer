import uvicorn
from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

import torch
import torch.nn as nn
from torchvision import models, transforms
from torchcam.methods import GradCAM
from torchcam.utils import overlay_mask

from PIL import Image
import numpy as np
import io
import base64

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "efficientnetb4_isic_best.pth"
CLASSES = ['Benign', 'Malignant']

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

model = None
cam_extractor = None


@app.on_event("startup")
def load_model():
    global model, cam_extractor
    model = models.efficientnet_b4(weights=None)
    num_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_features, 2)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()

    cam_extractor = GradCAM(model, target_layer="features.8")


def predict_image(image_bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img_input = val_transform(img).unsqueeze(0).to(DEVICE)

    outputs = model(img_input)
    probs = torch.softmax(outputs, dim=1)

    pred_idx = probs.argmax(1).item()
    pred_class = CLASSES[pred_idx]
    confidence = probs[0, pred_idx].item()

    # GradCAM
    cam = cam_extractor(pred_idx, outputs)[0].cpu().numpy()
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

    heatmap = Image.fromarray(np.uint8(cam * 255)).resize((224, 224))
    img_resized = img.resize((224, 224))

    overlay = overlay_mask(img_resized, heatmap, alpha=0.5)

    buffer = io.BytesIO()
    overlay.save(buffer, format="PNG")
    heatmap_base64 = base64.b64encode(buffer.getvalue()).decode()

    return pred_class, confidence, heatmap_base64


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        return JSONResponse({"error": "Invalid file type"}, status_code=400)

    contents = await file.read()

    try:
        pred, conf, heatmap = predict_image(contents)
    except Exception:
        return JSONResponse({"error": "Invalid image"}, status_code=400)

    return JSONResponse({
        "prediction": pred,
        "prediction_probability": f"{conf:.2%}",
        "heatmap": heatmap
    })
