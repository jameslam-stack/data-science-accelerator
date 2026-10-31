---
name: model-building
description: Build and train machine learning models. Includes classification, regression, model evaluation, hyperparameter tuning, and cross-validation.
allowed-tools:
  - Read
  - Write
  - Bash
metadata:
  stage: model-building
---
# Model Building Skill

## When to Use
- "Build a machine learning model"
- "Train a classifier/regressor"
- "Evaluate model performance"
- "Tune hyperparameters"
- "Cross-validate a model"
- "Improve model accuracy"

## Workflow Steps

### 1. Problem Setup
```python
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
import pandas as pd

# Split data into train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y if task == 'classification' else None
)

# Use a validation split or CV for reliable evaluation
```

### 2. Classification Models
```python
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

def train_classifier(X_train, y_train, model_type='random_forest'):
    if model_type == 'logistic':
        model = LogisticRegression(max_iter=1000)
    elif model_type == 'random_forest':
        model = RandomForestClassifier(n_estimators=100, random_state=42)
    elif model_type == 'gradient_boosting':
        model = GradientBoostingClassifier(random_state=42)
    elif model_type == 'svm':
        model = SVC(probability=True, random_state=42)
    else:
        model = RandomForestClassifier(random_state=42)

    model.fit(X_train, y_train)
    return model

# Evaluate
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
```

### 3. Regression Models
```python
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def train_regressor(X_train, y_train, model_type='random_forest'):
    if model_type == 'linear':
        model = LinearRegression()
    elif model_type == 'ridge':
        model = Ridge(alpha=1.0)
    elif model_type == 'lasso':
        model = Lasso(alpha=0.01)
    elif model_type == 'random_forest':
        model = RandomForestRegressor(n_estimators=100, random_state=42)
    elif model_type == 'gradient_boosting':
        model = GradientBoostingRegressor(random_state=42)
    else:
        model = RandomForestRegressor(random_state=42)

    model.fit(X_train, y_train)
    return model

# Evaluate
y_pred = model.predict(X_test)
print(f"RMSE: {mean_squared_error(y_test, y_pred, squared=False):.4f}")
print(f"MAE: {mean_absolute_error(y_test, y_pred):.4f}")
print(f"R2 Score: {r2_score(y_test, y_pred):.4f}")
```

### 4. Cross-Validation
```python
from sklearn.model_selection import cross_val_score, KFold

def cross_validate_model(model, X, y, cv=5, task='classification'):
    if task == 'classification':
        scoring = 'accuracy'
    else:
        scoring = 'r2'

    scores = cross_val_score(model, X, y, cv=cv, scoring=scoring)
    print(f"CV Scores: {scores}")
    print(f"Mean: {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")
    return scores
```

### 5. Hyperparameter Tuning
```python
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

def tune_hyperparameters(model, param_grid, X_train, y_train, cv=5, method='grid'):
    if method == 'grid':
        search = GridSearchCV(model, param_grid, cv=cv, scoring='accuracy', n_jobs=-1)
    else:
        search = RandomizedSearchCV(model, param_grid, n_iter=50, cv=cv,
                                    scoring='accuracy', random_state=42, n_jobs=-1)

    search.fit(X_train, y_train)
    print(f"Best params: {search.best_params_}")
    print(f"Best score: {search.best_score_:.4f}")
    return search.best_estimator_
```

### 6. Model Comparison
```python
def compare_models(X_train, y_train, X_test, y_test, models, task='classification'):
    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        if task == 'classification':
            score = accuracy_score(y_test, y_pred)
        else:
            score = r2_score(y_test, y_pred)
        results[name] = score
        print(f"{name}: {score:.4f}")
    return results
```

## Best Practices
- Always use a held-out test set for final evaluation
- Use cross-validation for reliable performance estimates
- Avoid data leakage — fit preprocessing on training data only
- Track experiments and hyperparameters
- Compare multiple models before selecting the best
- Use appropriate metrics for the task (accuracy, F1, RMSE, R2)

## Checklist
- [ ] Defined problem type (classification/regression)
- [ ] Split data into train/test
- [ ] Trained at least 2 model types
- [ ] Performed cross-validation
- [ ] Evaluated with appropriate metrics
- [ ] Compared models
- [ ] Selected best model