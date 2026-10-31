---
name: pycaret
description: PyCaret for low-code machine learning. Includes setup, model creation, comparison, tuning, ensembling, and deployment in a single workflow.
allowed-tools:
  - Read
  - Write
  - Bash
metadata:
  stage: pycaret
---
# PyCaret Skill

## When to Use
- "Build a model with low-code ML"
- "Compare multiple models quickly"
- "Automate model selection"
- "Set up an ML workflow fast"
- "Tune and deploy models with minimal code"

## Workflow Steps

### 1. Install PyCaret
```bash
pip install pycaret
# For classification
pip install pycaret[classification]
# For regression
pip install pycaret[regression]
# For time series
pip install pycaret[time_series]
```

### 2. Classification Setup
```python
from pycaret.classification import *

# Setup the experiment
clf = setup(
    data=df,
    target='target_column',
    session_id=42,
    train_size=0.8,
    fold=5,
    log_experiment=True,
    experiment_name='customer_churn',
    ignore_features=['id', 'customer_id'],
    preprocess=True,
    remove_outliers=True,
    outlier_threshold=0.05,
    fix_imbalance=True
)
```

### 3. Compare Models
```python
from pycaret.classification import *

# Compare all available models
best_model = compare_models(
    include=None,  # None = all models
    fold=5,
    sort='Accuracy',
    n_select=3
)
```

### 4. Create Specific Model
```python
from pycaret.classification import *

# Create a specific model
rf = create_model('rf')  # Random Forest
lr = create_model('lr')  # Logistic Regression
xgb = create_model('xgboost')

# Tune hyperparameters
tuned_rf = tune_model(rf, optimize='Accuracy', n_iter=50)
```

### 5. Ensemble Models
```python
from pycaret.classification import *

# Blend models
blend = blend_models(best_model_list)

# Stack models
stack = stack_models(best_model_list)

# Create ensemble with custom model
ensemble = ensemble_model(best_model_list, fold=5, choose_grid=False)
```

### 6. Evaluate Model
```python
from pycaret.classification import *

# Hold-out evaluation
evaluate_model(tuned_model)

# Predict on test set
predictions = predict_model(tuned_model, data=test_df)
```

### 7. Finalize Model
```python
from pycaret.classification import *

# Finalize model on full dataset
final_model = finalize_model(tuned_model)

# Save model
save_model(final_model, 'customer_churn_model')
```

### 8. Deployment
```python
from pycaret.classification import *

# Deploy to production
deploy_model(model=final_model, platform='AWS', authentication={
    'bucket': 'your-bucket',
    'key': 'your-key',
    'secret': 'your-secret'
})

# Or export for external serving
save_model(final_model, 'production_model.pkl')

# Load and predict
loaded = load_model('production_model.pkl')
predictions = predict_model(loaded, data=new_data)
```

### 9. Regression Example
```python
from pycaret.regression import *

reg = setup(
    data=df,
    target='price',
    session_id=42,
    train_size=0.8,
    fold=5,
    log_experiment=True
)

best_reg = compare_models(sort='R2')
tuned_reg = tune_model(best_reg, optimize='R2')
final_reg = finalize_model(tuned_reg)
```

### 10. Time Series Example
```python
from pycaret.time_series import *

ts_setup = setup(
    data=df,
    data_split=False,
    fh=12,  # forecast horizon
    session_id=42,
    fold=5,
    log_experiment=True
)

best_ts = compare_models()
final_ts = finalize_model(best_ts)
```

## PyCaret + MLflow Integration
```python
import mlflow
from pycaret.classification import *

# Enable automatic MLflow logging
mlflow.set_experiment("pycaret_experiment")

clf = setup(
    data=df,
    target='target',
    session_id=42,
    log_experiment=True,
    experiment_name="pycaret_mlflow"
)

# All models, metrics, and parameters are logged to MLflow automatically
best_model = compare_models()
tuned = tune_model(best_model)
finalized = finalize_model(tuned)

# Register in MLflow model registry
mlflow.register_model(
    model_uri="runs:/{run_id}/model",
    name="pycaret_model"
)
```

## Best Practices
- Always set a session_id for reproducibility
- Use cross-validation (fold=5) for reliable evaluation
- Compare models before tuning
- Log experiments for tracking and reproducibility
- Finalize model before deployment
- Use PyCaret's built-in preprocessing (or customize as needed)
- Combine with MLflow for experiment tracking

## Checklist
- [ ] Set up PyCaret session
- [ ] Compared multiple models
- [ ] Tuned best model
- [ ] Evaluated on hold-out set
- [ ] Ensembled if needed
- [ ] Finalized model
- [ ] Saved model artifact
- [ ] Tracked experiments with MLflow