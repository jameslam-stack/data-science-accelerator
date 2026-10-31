#!/usr/bin/env python3
import os
import json

# Create directory structure
skill_dirs = [
    '.claude/skills/data-cleaning',
    '.claude/skills/feature-engineering',
    '.claude/skills/feature-selection',
    '.claude/skills/model-building',
    '.claude/skills/model-saving',
    '.claude/skills/model-deployment'
]

for skill_dir in skill_dirs:
    os.makedirs(skill_dir, exist_ok=True)

# Data cleaning skill
data_cleaning_skill = '''---
name: data-cleaning
description: Data cleaning and preprocessing for ML pipelines. Handles missing values, duplicates, outliers, type fixes, text normalization.
allowed-tools:
  - Read
  - Write
  - Bash
metadata:
  stage: data-cleaning
---
# Data Cleaning Skill

## When to Use
- "Clean my dataset"
- "Preprocess data for ML"
- "Handle missing values"
- "Remove duplicates"
- "Fix data types"
- "Detect and remove outliers"
- "Normalize text fields"
- "Prepare data for ML models"

## Workflow Steps

### 1. Data Assessment
```python
import pandas as pd

def assess_data(df):
    """Comprehensive data assessment."""
    print("=== Data Assessment ===")
    print(f"Shape: {df.shape}")
    print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024:.2f} MB")
    \n    print("\\n=== Missing Values ===")
    missing = df.isnull().sum()
    missing_pct = (missing / len(df)) * 100
    print(pd.DataFrame({
        'Missing': missing.values,
        'Percentage': missing_pct.values,
        'Data Type': df.dtypes.values
    }))
    \n    print("\\n=== Duplicate Analysis ===")
    print(f"Total duplicates: {df.duplicated().sum()}")
    if df.duplicated().any():
        print(f"Columns with duplicates: {df.columns[df.duplicated()].tolist()}")
    \n    print("\\n=== Data Types ===")
    print(df.dtypes)
    \n    print("\\n=== Basic Statistics ===")
    print(df.describe(include='all'))
    \n    \n    return {\n        'missing_counts': missing.to_dict(),\n        'missing_percentage': missing_pct.to_dict(),\n        'duplicate_count': df.duplicated().sum(),\n        'shape': df.shape,\n        'dtypes': df.dtypes.to_dict()\n    }
```

### 2. Cleaning Pipeline Class
```python
class DataCleaningPipeline:
    def __init__(self, df, config=None):
        self.df = df.copy()
        self.config = config or CleaningConfig()
        self.cleaning_log = []
    
    def log(self, step, details=None):
        entry = {'step': step, 'details': details or {}}
        self.cleaning_log.append(entry)
        print(f"✓ {step}")
    
    def run_assessment(self):
        self.assessment = self.assess_data(self.df)
        return self
    
    def remove_duplicates(self, subset=None):
        if self.config.drop_duplicates:
            before = len(self.df)
            self.df = self.df.drop_duplicates(subset=subset)
            self.log('remove_duplicates', {'rows_before': before, 'rows_after': len(self.df)})
        return self
    
    def handle_missing_values(self):
        strategy = self.config.missing_value_strategy
        for col in self.df.columns:
            if self.df[col].isnull().any():
                if strategy == 'drop':
                    self.df = self.df.dropna(subset=[col])
                    self.log('drop_missing', {'column': col, 'strategy': 'drop'})
                elif strategy == 'mean' and pd.api.types.is_numeric_dtype(self.df[col]):
                    fill_val = self.df[col].mean()
                    self.df[col] = self.df[col].fillna(fill_val)
                    self.log('fill_missing', {'column': col, 'strategy': 'mean', 'value': fill_val})
                elif strategy == 'median' and pd.api.types.is_numeric_dtype(self.df[col]):
                    fill_val = self.df[col].median()
                    self.df[col] = self.df[col].fillna(fill_val)
                    self.log('fill_missing', {'column': col, 'strategy': 'median', 'value': fill_val})
                elif strategy == 'mode':
                    fill_val = self.df[col].mode()[0]
                    self.df[col] = self.df[col].fillna(fill_val)
                    self.log('fill_missing', {'column': col, 'strategy': 'mode', 'value': fill_val})
        return self
    
    def fix_data_types(self):
        for col, dtype in self.config.type_mapping.items():
            if col in self.df.columns:
                try:
                    if dtype == 'datetime':
                        self.df[col] = pd.to_datetime(self.df[col])
                    elif dtype == 'category':
                        self.df[col] = self.df[col].astype('category')
                    else:
                        self.df[col] = self.df[col].astype(dtype)
                    self.log('fix_data_types', {'column': col, 'target_dtype': dtype})
                except Exception as e:
                    self.log('type_error', {'column': col, 'error': str(e)})
        return self
    
    def normalize_text(self):
        text_cols = self.df.select_dtypes(include=['object']).columns
        for col in text_cols:
            self.df[col] = self.df[col].str.strip()
            if self.config.lowercase_text:
                self.df[col] = self.df[col].str.lower()
            self.log('normalize_text', {'column': col})
        return self
    
    def handle_outliers(self):
        if self.config.handle_outliers:
            numeric_cols = self.df.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                if col in self.config.outlier_columns:
                    Q1 = self.df[col].quantile(0.25)
                    Q3 = self.df[col].quantile(0.75)
                    IQR = Q3 - Q1
                    lower = Q1 - self.config.outlier_multiplier * IQR
                    upper = Q3 + self.config.outlier_multiplier * IQR
                    before = len(self.df)
                    self.df = self.df[(self.df[col] >= lower) & (self.df[col] <= upper)]
                    self.log('handle_outliers', {
                        'column': col,
                        'method': 'iqr',
                        'rows_removed': before - len(self.df)
                    })
        return self
    
    def validate_ranges(self):
        for col, (min_val, max_val) in self.config.ranges.items():
            if col in self.df.columns:
                invalid = ((self.df[col] < min_val) | (self.df[col] > max_val)).sum()
                if invalid > 0:
                    self.df = self.df[(self.df[col] >= min_val) & (self.df[col] <= max_val)]
                    self.log('validate_ranges', {'column': col, 'invalid_count': invalid})
        return self
    
    def run(self):
        self.run_assessment()
        self.remove_duplicates()
        self.handle_missing_values()
        self.fix_data_types()
        self.normalize_text()
        self.handle_outliers()
        self.validate_ranges()
        return self
    
    def get_results(self):
        return {
            'cleaned_data': self.df,
            'assessment': self.assessment,
            'cleaning_log': self.cleaning_log,
            'final_shape': self.df.shape
        }
```

### 3. Configuration
```python
class CleaningConfig:
    def __init__(self):
        self.drop_duplicates = True
        self.missing_value_strategy = 'auto'  # mean, median, mode, drop
        self.handle_outliers = True
        self.outlier_method = 'iqr'  # iqr, zscore
        self.outlier_multiplier = 1.5
        self.outlier_columns = ['age', 'salary']
        self.type_mapping = {}  # {'age': 'int64', 'date': 'datetime64'}
        self.ranges = {'age': (0, 120), 'salary': (0, 1000000)}
        self.lowercase_text = True
        self.drop_duplicates = True
        self.subset_for_duplicates = None
        self.numeric_fill_strategy = 'median'
        self.categorical_fill_strategy = 'mode'
'''

# Feature engineering skill
feature_engineering_skill = '''---
name: feature-engineering
description: Feature engineering and transformation for ML models. Includes encoding, scaling, creation, and selection.
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
def create_features(df):
    """Create new features from existing data."""
    df = df.copy()
    
    # Example 1: Interaction terms
    df['age_salary_interaction'] = df['age'] * df['salary']
    
    # Example 2: Polynomial features
    df['age_squared'] = df['age'] ** 2
    
    # Example 3: Temporal features
    if 'date' in df.columns:
        df['year'] = pd.to_datetime(df['date']).dt.year
        df['month'] = pd.to_datetime(df['date']).dt.month
        df['day_of_week'] = pd.to_datetime(df['date']).dt.dayofweek
    
    # Example 4: Ratio features
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
from sklearn.preprocessing import StandardScaler, MinMaxScaler

def scale_features(df, scaler_type='standard'):
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    if scaler_type == 'standard':
        scaler = StandardScaler()
    elif scaler_type == 'minmax':
        scaler = MinMaxScaler()
    else:
        scaler = StandardScaler()
    
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    return df
```

### 4. Feature Selection
```python
def select_features(df, method='all', k=10):
    if method == 'all':
        return df
    
    elif method == 'correlation':
        # Simple correlation-based selection
        corr_matrix = df.corr()
        return df.select_columns([col for col in corr_matrix.columns if abs(corr_matrix[col].mean()) > 0.1])
    
    elif method == 'k_best':
        from sklearn.feature_selection import SelectKBest, f_classif
        selector = SelectKBest(f_classif, k=k)
        selected_cols = selector.get_support(indices=True)
        return df.iloc[:, selected_cols]
    
    return df
EOF
""", interval=0.1)