---
name: feature-engineering
description: Feature engineering and transformation for ML models. Includes encoding, scaling, feature creation, and selection.
allowed-tools:
  - Read
  - Write
  - Bash
metadata:
  stage: feature-engineering
---
# Feature Engineering Skill

## When to Use
- "Create features for ML"
- "Engineer features from raw data"
- "Transform categorical variables"
- "Scale numerical features"
- "Create interaction terms"
- "Generate ML-ready features"

## Workflow Steps

### 1. Feature Creation
```python
import pandas as pd
import numpy as np

def create_features(df):
    df = df.copy()

    # Interaction terms
    df['age_salary_interaction'] = df['age'] * df['salary']

    # Polynomial features
    df['age_squared'] = df['age'] ** 2

    # Temporal features
    if 'date' in df.columns:
        df['year'] = pd.to_datetime(df['date']).dt.year
        df['month'] = pd.to_datetime(df['date']).dt.month
        df['day_of_week'] = pd.to_datetime(df['date']).dt.dayofweek

    # Ratio features
    if 'salary' in df.columns and 'age' in df.columns:
        df['salary_per_year'] = df['salary'] / (df['age'] + 1)

    return df
```

### 2. Encoding Techniques
```python
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

def encode_categorical(df, columns=None, method='label_encoder'):
    if columns is None:
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    else:
        categorical_cols = columns

    if method == 'label_encoder':
        le = LabelEncoder()
        df[categorical_cols] = df[categorical_cols].apply(le.fit_transform)
        return df

    elif method == 'onehot':
        ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        categorical_data = df[categorical_cols]
        encoded = ohe.fit_transform(categorical_data)
        feature_names = ohe.get_feature_names_out(categorical_cols)
        encoded_df = pd.DataFrame(encoded, columns=feature_names)
        return pd.concat([df.drop(columns=categorical_cols), encoded_df], axis=1)

    return df
```

### 3. Scaling and Normalization
```python
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

def scale_features(df, scaler_type='standard'):
    numeric_cols = df.select_dtypes(include=[np.number]).columns

    if scaler_type == 'standard':
        scaler = StandardScaler()
    elif scaler_type == 'minmax':
        scaler = MinMaxScaler()
    elif scaler_type == 'robust':
        scaler = RobustScaler()
    else:
        scaler = StandardScaler()

    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    return df

# Or save the scaler for later use
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)  # Use same scaler
```

### 4. Feature Selection
```python
def select_features(df, method='all', k=10):
    if method == 'all':
        return df

    elif method == 'correlation':
        corr_matrix = df.corr()
        return df[[col for col in corr_matrix.columns if abs(corr_matrix[col].mean()) > 0.1]]

    elif method == 'k_best':
        from sklearn.feature_selection import SelectKBest, f_classif
        selector = SelectKBest(f_classif, k=k)
        selected_cols = selector.get_support(indices=True)
        return df.iloc[:, selected_cols]

    return df
```

## Best Practices
- Avoid data leakage: fit transformers on training data only
- Log feature engineering steps
- Save scaler/encoder objects for inference
- Check for multicollinearity
- Create domain-specific features

## Checklist
- [ ] Assessed feature distributions
- [ ] Created interaction features
- [ ] Encoded categorical variables
- [ ] Scaled numerical features
- [ ] Performed feature selection
- [ ] Checked for data leakage
- [ ] Saved transformation objects
