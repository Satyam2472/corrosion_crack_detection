import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import json
import sys
import os

# ─────────────────────────────────────────
# LOAD CONFIG + WEIGHTS  (done once)
# ─────────────────────────────────────────
MODEL_DIR    = "model"
WEIGHTS_PATH = os.path.join(MODEL_DIR, "resnet18_corrosion.pth")
CONFIG_PATH  = os.path.join(MODEL_DIR, "model_config.json")

def load_model():
    with open(CONFIG_PATH) as f:
        config = json.load(f)

    model = models.resnet18(pretrained=False)       # architecture only, no pretrained weights
    # model.fc = nn.Linear(512, config["num_classes"])
    model.fc = nn.Sequential(
        nn.Dropout(p=0.5),
        nn.Linear(512, config["num_classes"])
    )
    model.load_state_dict(torch.load(WEIGHTS_PATH, map_location="cpu"))
    model.eval()                                    # inference mode — disables dropout/batchnorm train behavior

    transform = transforms.Compose([
        transforms.Resize((config["input_size"], config["input_size"])),
        transforms.ToTensor(),
        transforms.Normalize(config["normalize_mean"], config["normalize_std"])
    ])

    return model, transform, config["class_names"]

# ─────────────────────────────────────────
# PREDICT SINGLE IMAGE
# ─────────────────────────────────────────
def predict(image_path, model, transform, class_names):
    image  = Image.open(image_path).convert("RGB")
    tensor = transform(image).unsqueeze(0)          # add batch dimension → [1, 3, 224, 224]

    with torch.no_grad():
        outputs     = model(tensor)
        probs       = torch.softmax(outputs, dim=1)[0]   # convert logits → probabilities
        confidence, predicted_idx = probs.max(0)

    label      = class_names[predicted_idx.item()]
    confidence = confidence.item() * 100

    # Print all class probabilities
    print(f"\nImage     : {image_path}")
    print(f"Prediction: {label}  ({confidence:.1f}% confidence)")
    print("─" * 40)
    for i, name in enumerate(class_names):
        bar = "█" * int(probs[i].item() * 30)
        print(f"  {name:<12} {probs[i].item()*100:5.1f}%  {bar}")

    return label, confidence

# ─────────────────────────────────────────
# RUN
# ─────────────────────────────────────────
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python inference.py path/to/image.jpg")
        print("       python inference.py path/to/folder/")
        sys.exit(1)

    model, transform, class_names = load_model()
    print(f"✅ Model loaded — classes: {class_names}")

    input_path = sys.argv[1]

    # Single image
    if os.path.isfile(input_path):
        predict(input_path, model, transform, class_names)

    # Folder — run on all images inside
    elif os.path.isdir(input_path):
        extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        images     = [f for f in os.listdir(input_path)
                      if os.path.splitext(f)[1].lower() in extensions]
        print(f"\nFound {len(images)} images in {input_path}\n")
        for img_file in images:
            predict(os.path.join(input_path, img_file), model, transform, class_names)
    else:
        print(f"Path not found: {input_path}")