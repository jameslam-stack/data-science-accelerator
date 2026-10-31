---
name: mlflow
description: MLflow for experiment tracking, model registry, and MLOps. Includes logging metrics, parameters, artifacts, and model lifecycle management.
allowed-tools:
  - Read
  - Write
  - Bash
metadata:
  stage: mlflow
---
# MLflow Skill

## When to Use
- "Track ML experiments"
- "Log metrics and parameters"
- "Register a model"
- "Set up MLflow tracking server"
- "Compare experiments"
- "Manage model lifecycle"

## Workflow Steps

### 1. Install MLflow
```bash
pip install mlflow
# For model serving
pip install mlflow[extras]
```

### 2. Basic Experiment Tracking
```python
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

mlflow.set_experiment("Random Forest Experiment")

with mlflow.start_run():
    # Log parameters
    mlflow.log_param("n_estimators", 100)
    mlflow.log_param("max_depth", 5)
    mlflow.log_param("random_state", 42)

    # Train model
    model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    model.fit(X_train, y_train)

    # Log metrics
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    mlflow.log_metric("accuracy", accuracy)

    # Log model
    mlflow.sklearn.log_model(model, "model")

    # Log artifacts
    mlflow.log_artifact("preprocessing.pkl")
```

### 3. Log Multiple Metrics and Params
```python
import mlflow

with mlflow.start_run(run_name="Experiment 1"):
    # Log multiple parameters
    params = {"learning_rate": 0.01, "epochs": 100, "batch_size": 32}
    mlflow.log_params(params)

    # Log multiple metrics
    metrics = {"accuracy": 0.95, "precision": 0.93, "recall": 0.94, "f1_score": 0.935}
    mlflow.log_metrics(metrics)

    # Log model with signature
    mlflow.sklearn.log_model(model, "model", signature=signature)
```

### 4. Model Registry
```python
import mlflow
import mlflow.sklearn

# Register a model
model_uri = "runs:/<run-id>/model"
model_name = "CustomerChurnModel"

mlflow.register_model(model_uri, model_name)

# Transition model stage
client = mlflow.tracking.MlflowClient()
client.transition_model_version_stage(
    name=model_name,
    version=1,
    stage="Production",
    archive_existing_versions=True
)
```

### 5. Load and Serve Model
```python
import mlflow
import pandas as pd

# Load registered model
model = mlflow.sklearn.load_model("models:/CustomerChurnModel/Production")

# Predict
predictions = model.predict(X_test)

# Serve with MLflow
# Run: mlflow models serve -m "models:/CustomerChurnModel/Production" -p 5000
```

### 6. Hyperparameter Tuning with MLflow
```python
from sklearn.model_selection import GridSearchCV
import mlflow

def tune_with_mlflow(model, param_grid, X_train, y_train):
    with mlflow.start_run(run_name="Hyperparameter Tuning"):
        grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy')
        grid_search.fit(X_train, y_train)

        # Log all combinations
        for i, params in enumerate(grid_search.cv_results_['params']):
            with mlflow.start_run(run_name=f"Trial {i}", nested=True):
                mlflow.log_params(params)
                mlflow.log_metric("mean_score", grid_search.cv_results_['mean_test_score'][i])

        # Log best model
        mlflow.log_params(grid_search.best_params_)
        mlflow.log_metric("best_score", grid_search.best_score_)
        mlflow.sklearn.log_model(grid_search.best_estimator_, "best_model")
```

### 7. Tracking Server (Production)
```bash
# Start tracking server
mlflow server --host 0.0.0.0 --port 5000 --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./artifacts

# In code
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("production-experiment")
```

## MLflow + PyCaret Integration
```python
import mlflow
from pycaret.classification import *

# Setup PyCaret
clf = setup(data, target='target', session_id=42, log_experiment=True,
            experiment_name='pycaret_mlflow')

# MLflow logs automatically when PyCaret is configured
model = create_model('rf')

# Compare models
best = compare_models()
```

## Best Practices
- Use meaningful experiment names and run tags
- Log all hyperparameters and metrics consistently
- Use the model registry for production deployments
- Enable model versioning and stage transitions
- Set up a persistent tracking server for teams
- Log preprocessing artifacts alongside models
- Use nested runs for hyperparameter tuning

## Checklist
- [ ] Set up MLflow tracking
- [ ] Logged experiment parameters
- [ ] Logged evaluation metrics
- [ ] Logged model artifact
- [ ] Registered model in registry
- [ ] Set up model versioning
- [ ] Documented experiment results