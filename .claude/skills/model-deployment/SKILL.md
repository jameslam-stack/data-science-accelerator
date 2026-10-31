---
name: model-deployment
description: Deploy machine learning models to production. Includes API serving, containerization, batch inference, and monitoring.
allowed-tools:
  - Read
  - Write
  - Bash
metadata:
  stage: model-deployment
---
# Model Deployment Skill

## When to Use
- "Deploy a model as an API"
- "Serve predictions"
- "Containerize a model"
- "Set up batch inference"
- "Monitor model performance"

## Workflow Steps

### 1. Save Model and Preprocessing Pipeline
```python
import joblib
import pickle

# Save model and preprocessing objects together
def save_artifacts(model, scaler, encoder, path='./'):
    joblib.dump(model, f'{path}model.pkl')
    joblib.dump(scaler, f'{path}scaler.pkl')
    if encoder is not None:
        joblib.dump(encoder, f'{path}encoder.pkl')
    print("Artifacts saved successfully")

# Load artifacts for inference
def load_artifacts(path='./'):
    model = joblib.load(f'{path}model.pkl')
    scaler = joblib.load(f'{path}scaler.pkl')
    encoder = None
    try:
        encoder = joblib.load(f'{path}encoder.pkl')
    except FileNotFoundError:
        pass
    return model, scaler, encoder
```

### 2. FastAPI Prediction Endpoint
```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI(title="ML Model API")

# Load model on startup
model = joblib.load('model.pkl')
scaler = joblib.load('scaler.pkl')

class PredictionInput(BaseModel):
    feature1: float
    feature2: float
    feature3: float

@app.post("/predict")
def predict(input_data: PredictionInput):
    try:
        features = np.array([[input_data.feature1, input_data.feature2, input_data.feature3]])
        features_scaled = scaler.transform(features)
        prediction = model.predict(features_scaled)
        return {"prediction": float(prediction[0])}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
def health_check():
    return {"status": "healthy"}

# Run: uvicorn main:app --reload
```

### 3. Batch Inference
```python
import pandas as pd
import joblib

def batch_predict(input_csv, output_csv, model_path='model.pkl', scaler_path='scaler.pkl'):
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)

    df = pd.read_csv(input_csv)
    X_scaled = scaler.transform(df)
    predictions = model.predict(X_scaled)

    df['prediction'] = predictions
    df.to_csv(output_csv, index=False)
    print(f"Predictions saved to {output_csv}")
    return df
```

### 4. Docker Containerization
```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY model.pkl scaler.pkl .
COPY app.py .

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
```

```bash
# Build and run
docker build -t ml-model-api .
docker run -p 8000:8000 ml-model-api
```

### 5. Model Monitoring
```python
import time
from datetime import datetime

class ModelMonitor:
    def __init__(self):
        self.predictions = []
        self.errors = []

    def log_prediction(self, input_data, prediction, latency_ms):
        self.predictions.append({
            'timestamp': datetime.now().isoformat(),
            'prediction': prediction,
            'latency_ms': latency_ms
        })

    def log_error(self, error):
        self.errors.append({
            'timestamp': datetime.now().isoformat(),
            'error': str(error)
        })

    def get_metrics(self):
        if not self.predictions:
            return {"total_predictions": 0}
        latencies = [p['latency_ms'] for p in self.predictions]
        return {
            "total_predictions": len(self.predictions),
            "total_errors": len(self.errors),
            "avg_latency_ms": sum(latencies) / len(latencies),
            "max_latency_ms": max(latencies)
        }
```

## Best Practices
- Version models and keep deployment artifacts
- Use environment variables for configuration
- Implement health checks and graceful shutdown
- Log predictions and errors for monitoring
- Set up alerts for model drift and performance degradation
- Use CI/CD pipelines for automated deployment
- Scale horizontally with load balancers

## Checklist
- [ ] Saved model and preprocessing artifacts
- [ ] Created API endpoint with health check
- [ ] Tested predictions locally
- [ ] Containerized with Docker
- [ ] Set up logging and monitoring
- [ ] Documented deployment process