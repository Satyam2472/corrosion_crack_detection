import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
import matplotlib.pyplot as plt
import time
import json
from torchvision.models import ResNet18_Weights

# ─────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────
DATA_DIR    = "pipeline_corrosion_dataset"
BATCH_SIZE  = 32
EPOCHS      = 30
LR          = 1e-4
NUM_CLASSES = 2
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Output paths
MODEL_DIR   = "model"
os.makedirs(MODEL_DIR, exist_ok=True)

WEIGHTS_PATH  = os.path.join(MODEL_DIR, "resnet18_corrosion.pth")   # weights only
CONFIG_PATH   = os.path.join(MODEL_DIR, "model_config.json")        # metadata

print(f"Using device: {DEVICE}")

# ─────────────────────────────────────────
# TRANSFORMS
# ─────────────────────────────────────────
train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(30),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.4),
    transforms.RandomGrayscale(p=0.1),
    transforms.GaussianBlur(kernel_size=3),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

test_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ─────────────────────────────────────────
# DATASETS
# ─────────────────────────────────────────
train_dataset = datasets.ImageFolder(os.path.join(DATA_DIR, "train"), transform=train_transforms)
test_dataset  = datasets.ImageFolder(os.path.join(DATA_DIR, "test"),  transform=test_transforms)

train_loader  = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0, pin_memory=True)
test_loader   = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)

CLASS_NAMES = train_dataset.classes   # ['Corroded', 'Normal'] — alphabetical
print(f"Classes: {CLASS_NAMES}")

# ─────────────────────────────────────────
# MODEL
# ─────────────────────────────────────────
model = models.resnet18(weights=ResNet18_Weights.DEFAULT)

for param in model.parameters():       # freeze backbone — Phase 1
    param.requires_grad = False

model.fc = nn.Sequential(
    nn.Dropout(p=0.5),       # randomly drops 50% of neurons
    nn.Linear(512, NUM_CLASSES)
)
model     = model.to(DEVICE)

# ─────────────────────────────────────────
# LOSS / OPTIMIZER / SCALER
# ─────────────────────────────────────────
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.fc.parameters(), lr=LR)
scaler = torch.amp.GradScaler('cuda')
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

# ─────────────────────────────────────────
# TRAIN / EVAL HELPERS
# ─────────────────────────────────────────
def train_one_epoch(model, loader, optimizer, criterion, scaler):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        with autocast():
            outputs = model(images)
            loss    = criterion(outputs, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        running_loss += loss.item() * images.size(0)
        _, predicted  = outputs.max(1)
        correct      += predicted.eq(labels).sum().item()
        total        += labels.size(0)
    return running_loss / total, 100. * correct / total

def evaluate(model, loader, criterion):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs  = model(images)
            loss     = criterion(outputs, labels)
            running_loss += loss.item() * images.size(0)
            _, predicted  = outputs.max(1)
            correct      += predicted.eq(labels).sum().item()
            total        += labels.size(0)
    return running_loss / total, 100. * correct / total

# ─────────────────────────────────────────
# TRAINING LOOP
# ─────────────────────────────────────────
history      = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
best_val_acc = 0.0
UNFREEZE_EPOCH = 10

print("\n── Phase 1: Training classifier head only ──")

for epoch in range(1, EPOCHS + 1):

    if epoch == UNFREEZE_EPOCH + 1:
        print("\n── Phase 2: Unfreezing full network ──")
        for param in model.parameters():
            param.requires_grad = True
        optimizer = optim.Adam(model.parameters(), lr=LR * 0.1)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    t0 = time.time()
    train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, scaler)
    val_loss,   val_acc   = evaluate(model, test_loader, criterion)
    scheduler.step()

    history["train_loss"].append(train_loss)
    history["train_acc"].append(train_acc)
    history["val_loss"].append(val_loss)
    history["val_acc"].append(val_acc)

    if val_acc > best_val_acc:
        best_val_acc = val_acc

        # ── Save weights only (recommended) ──
        torch.save(model.state_dict(), WEIGHTS_PATH)
        saved = "✓ saved"
    else:
        saved = ""

    print(f"Epoch [{epoch:02d}/{EPOCHS}]  "
          f"Train Loss: {train_loss:.4f}  Acc: {train_acc:.1f}%  |  "
          f"Val Loss: {val_loss:.4f}  Acc: {val_acc:.1f}%  "
          f"[{time.time()-t0:.1f}s]  {saved}")

# ─────────────────────────────────────────
# SAVE CONFIG — needed to reload model later
# ─────────────────────────────────────────
config = {
    "architecture": "resnet18",
    "num_classes":  NUM_CLASSES,
    "class_names":  CLASS_NAMES,      # e.g. ['Corroded', 'Normal']
    "input_size":   224,
    "best_val_acc": round(best_val_acc, 2),
    "normalize_mean": [0.485, 0.456, 0.406],
    "normalize_std":  [0.229, 0.224, 0.225],
}
with open(CONFIG_PATH, "w") as f:
    json.dump(config, f, indent=2)

print(f"\n✅ Best Val Accuracy : {best_val_acc:.1f}%")
print(f"✅ Weights saved     : {WEIGHTS_PATH}")
print(f"✅ Config saved      : {CONFIG_PATH}")

# ─────────────────────────────────────────
# PLOT
# ─────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
ax1.plot(history["train_loss"], label="Train"); ax1.plot(history["val_loss"], label="Val")
ax1.set_title("Loss"); ax1.legend()
ax2.plot(history["train_acc"],  label="Train"); ax2.plot(history["val_acc"],  label="Val")
ax2.set_title("Accuracy (%)"); ax2.legend()
plt.tight_layout()
plt.savefig(os.path.join(MODEL_DIR, "training_curves.png"), dpi=150)
print(f"✅ Curves saved      : {MODEL_DIR}/training_curves.png")