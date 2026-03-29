## CorroScan — Pipeline Corrosion Detection

CorroScan is a lightweight computer-vision project that detects corrosion on pipeline surface images. It includes:
- A FastAPI backend for inference
- A simple HTML/CSS/JS frontend for uploading images and viewing predictions
- A training script based on ResNet-18

## Project Structure
- `app.py` FastAPI server that loads the model once and serves predictions
- `inference.py` CLI tool for predicting single images or folders
- `train.py` Training script that creates model weights and config
- `index.html` Frontend UI that calls the API
- `model/` Model weights and config (`resnet18_corrosion.pth`, `model_config.json`)
- `pipeline_corrosion_dataset/` Training data (train/test folders)

## Setup
1. Create and activate a virtual environment
2. Install dependencies

```bash
pip install -r requirements.txt
```

## Run The API
Start the FastAPI server with Uvicorn:

```bash
uvicorn app:app --host 0.0.0.0 --port 8000
```

Health check:

```bash
curl http://localhost:8000/health
```

## Run The Frontend
Open `index.html` in a browser. The frontend expects the API at:

```
http://localhost:8000
```

Drop an image and click **RUN ANALYSIS**.

## CLI Inference
Predict a single image:

```bash
python inference.py path/to/image.jpg
```

Predict all images in a folder:

```bash
python inference.py path/to/folder/
```

## Training
Make sure your dataset is structured like:

```
pipeline_corrosion_dataset/
  train/
    Corroded/
    Normal/
  test/
    Corroded/
    Normal/
```

Then run:

```bash
python train.py
```

This will save:
- `model/resnet18_corrosion.pth`
- `model/model_config.json`
- `model/training_curves.png`

## API Endpoints
- `GET /` Model info and classes
- `GET /health` Health status
- `POST /predict` Multipart image upload
- `POST /predict-base64` Base64 JSON payload

Example `POST /predict` with curl:

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: multipart/form-data" \
  -F "file=@path/to/image.jpg"
```

## Notes
- The API requires `model/resnet18_corrosion.pth` and `model/model_config.json`.
- The model is loaded once at startup for fast inference.
