---
name: feature-selection
description: Feature selection and dimensionality reduction for ML models. Includes correlation, importance-based, and wrapper methods.
allowed-tools:
  - Read
  - Write
  - Bash
metadata:
  stage: feature-selection
---
# Feature Selection Skill

## When to Use
- "Select best features for model"
- "Reduce dimensionality"
- "Remove correlated features"
- "Find important features"
- "Improve model performance with feature selection"

## Workflow Steps

### 1. Correlation-Based Selection
```python
import pandas as pd
import numpy as np

def select_by_correlation(df, threshold=0.95):
    """Remove highly correlated features."""
    corr_matrix = df.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
    return df.drop(columns=to_drop)
```

### 2. Importance-Based Selection
```python
from sklearn.feature_selection import SelectKBest, f_classif, f_regression

def select_k_best(X, y, k=10, task='classification'):
    if task == 'classification':
        score_func = f_classif
    else:
        score_func = f_regression

    selector = SelectKBest(score_func, k=k)
    X_selected = selector.fit_transform(X, y)
    selected_features = X.columns[selector.get_support()].tolist()
    return X_selected, selected_features
```

### 3. Wrapper Methods
```python
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier

def recursive_feature_elimination(X, y, n_features=10):
    model = RandomForestClassifier()
    selector = RFE(model, n_features_to_select=n_features)
    X_selected = selector.fit_transform(X, y)
    selected_features = X.columns[selector.support_].tolist()
    return X_selected, selected_features
```

### 4. Embedded Methods
```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Lasso

def select_by_importance(X, y, model_type='random_forest', threshold=0.01):
    if model_type == 'random_forest':
        model = RandomForestClassifier()
    elif model_type == 'lasso':
        model = Lasso(alpha=0.01)
    else:
        model = RandomForestClassifier()

    model.fit(X, y)
    importance = model.feature_importances_ if hasattr(model, 'feature_importances_') else model.coef_
    selected_features = X.columns[importance > threshold].tolist()
    return X[selected_features], selected_features
```

## Best Practices
- Use cross-validation for feature selection
- Avoid data leakage
- Consider computational cost
- Balance interpretability vs performance
- Validate on holdout set

## Checklist
- [ ] Assessed feature distributions
- [ ] Removed highly correlated features
- [ ] Selected top K features
- [ ] Validated on holdout set
- [ ] Documented selection process
