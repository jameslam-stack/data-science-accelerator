---
name: model-saving
description: Save and load machine learning models and preprocessing artifacts. Includes joblib, pickle, ONNX, and model versioning.
allowed-tools:
  - Read
  - Write
  - Bash
metadata:
  stage: model-saving
---
# Model Saving Skill

## When to Use
- "Save a trained model"
- "Load a model for inference"
- "Export model to ONNX"
- "Version a model"
- "Package model artifacts"

## Workflow Steps

### 1. Save with joblib (Recommended)
```python
import joblib
from sklearn.ensemble import RandomForestClassifier

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save model
joblib.dump(model, 'model.pkl')
print(f"Model saved to model.pkl ({os.path.getsize('model.pkl') / 1024:.2f} KB)")

# Load model
loaded_model = joblib.load('model.pkl')
```

### 2. Save Model + Preprocessing Pipeline
```python
import joblib
import os

def save_model_artifacts(model, scaler, encoder, feature_names, path='./models/'):
    os.makedirs(path, exist_ok=True)

    artifacts = {
        'model': model,
        'scaler': scaler,
        'encoder': encoder,
        'feature_names': feature_names,
        'version': '1.0.0'
    }

    joblib.dump(artifacts, os.path.join(path, 'artifacts.pkl'))
    print(f"Artifacts saved to {path}artifacts.pkl")

def load_model_artifacts(path='./models/'):
    artifacts = joblib.load(os.path.join(path, 'artifacts.pkl'))
    return artifacts['model'], artifacts['scaler'], artifacts['encoder'], artifacts['feature_names']
```

### 3. Save with pickle
```python
import pickle

# Save
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)

# Load
with open('model.pkl', 'rb') as f:
    loaded_model = pickle.load(f)
```

### 4. Export to ONNX
```python
from sklearn.linear_model import LogisticRegression
from onnxmltools import convert_sklearn
from onnx import helper

# Convert sklearn model to ONNX
def convert_to_onnx(model, X_sample, output_path='model.onnx'):
    from onnxmltools.utils import calc_feature_range

    # Convert
    onnx_model = convert_sklearn(model, name='Model', feature_names=X_sample.columns.tolist())
    onnx_model = helper.make_model(onnx_model)

    with open(output_path, 'wb') as f:
        f.write(onnx_model.SerializeToString())
    print(f"ONNX model saved to {output_path}")

# Load ONNX for inference
import onnxruntime as ort

def load_onnx_model(path='model.onnx'):
    return ort.InferenceSession(path)

# Predict with ONNX
session = load_onnx_model()
input_name = session.get_inputs()[0].name
output = session.run(None, {input_name: X_test.values.astype(np.float32)})
```

### 5. Model Versioning
```python
import joblib
import os
from datetime import datetime

def save_versioned_model(model, version_dir='./models/'):
    os.makedirs(version_dir, exist_ok=True)

    version = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f'model_v{version}.pkl'
    filepath = os.path.join(version_dir, filename)

    joblib.dump(model, filepath)
    print(f"Model saved as {filename}")

    # Save metadata
    metadata = {
        'version': version,
        'filename': filename,
        'created_at': datetime.now().isoformat(),
        'model_type': type(model).__name__
    }
    joblib.dump(metadata, os.path.join(version_dir, 'latest_metadata.pkl'))
    return filepath

def load_latest_model(version_dir='./models/'):
    files = [f for f in os.listdir(version_dir) if f.startswith('model_v') and f.endswith('.pkl')]
    if not files:
        raise FileNotFoundError("No saved models found")
    latest = max(files)
    return joblib.load(os.path.join(version_dir, latest))
```

### 6. Model Registry Pattern
```python
class ModelRegistry:
    def __init__(self, registry_dir='./model_registry/'):
        self.registry_dir = registry_dir
        os.makedirs(registry_dir, exist_ok=True)

    def register(self, model, name, version, metadata=None):
        path = os.path.join(self.registry_dir, f'{name}_{version}.pkl')
        joblib.dump(model, path)

        meta = {
            'name': name,
            'version': version,
            'path': path,
            'created_at': datetime.now().isoformat(),
            'metadata': metadata or {}
        }
        joblib.dump(meta, os.path.join(self.registry_dir, f'{name}_{version}_meta.pkl'))
        return path

    def load(self, name, version=None):
        if version is None:
            # Load latest
            files = [f for f in os.listdir(self.registry_dir) if f.startswith(f'{name}_') and f.endswith('.pkl')]
            if not files:
                raise FileNotFoundError(f"No model named {name} found")
            version = max(files).split('_')[-1].replace('.pkl', '')
        path = os.path.join(self.registry_dir, f'{name}_{version}.pkl')
        return joblib.load(path)
```

## Best Practices
- Use joblib over pickle for large scikit-learn models (better compression)
- Always save preprocessing artifacts alongside the model
- Version models and keep a changelog
- Test loaded models produce identical predictions
- Use ONNX for cross-platform deployment
- Store models in a registry with metadata
- Compress large models for storage

## Checklist
- [ ] Saved model with joblib
- [ ] Saved preprocessing artifacts (scaler, encoder)
- [ ] Tested model loads correctly
- [ ] Verified predictions match training-time output
- [ ] Versioned the model
- [ ] Exported to ONNX if needed
- [ ] Documented model metadata