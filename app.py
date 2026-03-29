import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import json
import os
import io
import base64

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ─────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────
MODEL_DIR    = "model"
WEIGHTS_PATH = os.path.join(MODEL_DIR, "resnet18_corrosion.pth")
CONFIG_PATH  = os.path.join(MODEL_DIR, "model_config.json")

# ─────────────────────────────────────────
# LOAD MODEL ONCE AT STARTUP
# This is the key — model loads into memory
# once and stays there for all requests
# ─────────────────────────────────────────
print("Loading model...")

with open(CONFIG_PATH) as f:
    config = json.load(f)

DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASS_NAMES = config["class_names"]

model = models.resnet18(pretrained=False)
model.fc = nn.Sequential(
        nn.Dropout(p=0.5),
        nn.Linear(512, config["num_classes"])
    )
model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()                             # critical — keeps batchnorm/dropout in eval mode

transform = transforms.Compose([
    transforms.Resize((config["input_size"], config["input_size"])),
    transforms.ToTensor(),
    transforms.Normalize(config["normalize_mean"], config["normalize_std"])
])

print(f"✅ Model ready on {DEVICE}  |  Classes: {CLASS_NAMES}")

# ─────────────────────────────────────────
# FASTAPI APP
# ─────────────────────────────────────────
app = FastAPI(
    title="Corrosion Detector API",
    description="Binary classifier — Corroded vs Normal pipeline surfaces",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],        # lock this down in production
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────────────────────────────────────
# RESPONSE SCHEMA
# ─────────────────────────────────────────
class PredictionResponse(BaseModel):
    prediction:  str
    confidence:  float
    probabilities: dict

# ─────────────────────────────────────────
# INFERENCE HELPER
# ─────────────────────────────────────────
def run_inference(image: Image.Image) -> PredictionResponse:
    tensor = transform(image).unsqueeze(0).to(DEVICE)   # [1, 3, 224, 224]

    with torch.no_grad():
        outputs = model(tensor)
        probs   = torch.softmax(outputs, dim=1)[0]

    confidence, predicted_idx = probs.max(0)
    label = CLASS_NAMES[predicted_idx.item()]

    return PredictionResponse(
        prediction   = label,
        confidence   = round(confidence.item() * 100, 2),
        probabilities= {name: round(probs[i].item() * 100, 2)
                        for i, name in enumerate(CLASS_NAMES)}
    )

# ─────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────
@app.get("/")
def root():
    return {
        "status":  "running",
        "model":   config["architecture"],
        "classes": CLASS_NAMES,
        "best_val_acc": config["best_val_acc"]
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    """Upload an image file and get corrosion prediction."""

    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    contents = await file.read()
    image    = Image.open(io.BytesIO(contents)).convert("RGB")

    return run_inference(image)

@app.post("/predict-base64", response_model=PredictionResponse)
async def predict_base64(payload: dict):
    """Send image as base64 string — useful for frontends."""

    try:
        image_data = base64.b64decode(payload["image"])
        image      = Image.open(io.BytesIO(image_data)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid base64 image")

    return run_inference(image)

@app.get("/health")
def health():
    return {"status": "ok", "device": str(DEVICE)}