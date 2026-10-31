---
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
    print("=== Data Assessment ===")
    print(f"Shape: {df.shape}")
    print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024:.2f} MB")

    print("\n=== Missing Values ===")
    missing = df.isnull().sum()
    missing_pct = (missing / len(df)) * 100
    print(pd.DataFrame({
        'Missing': missing.values,
        'Percentage': missing_pct.values,
        'Data Type': df.dtypes.values
    }))

    print("\n=== Duplicate Analysis ===")
    print(f"Total duplicates: {df.duplicated().sum()}")
    if df.duplicated().any():
        print(f"Columns with duplicates: {df.columns[df.duplicated()].tolist()}")

    print("\n=== Data Types ===")
    print(df.dtypes)

    print("\n=== Basic Statistics ===")
    print(df.describe(include='all'))

    return {
        'missing_counts': missing.to_dict(),
        'missing_percentage': missing_pct.to_dict(),
        'duplicate_count': df.duplicated().sum(),
        'shape': df.shape,
        'dtypes': df.dtypes.to_dict()
    }
```

### 2. Cleaning Pipeline
```python
class DataCleaningPipeline:
    def __init__(self, df, config=None):
        self.df = df.copy()
        self.config = config or CleaningConfig()
        self.cleaning_log = []

    def log(self, step, details=None):
        entry = {'step': step, 'details': details or {}}
        self.cleaning_log.append(entry)
        print(f"OK {step}")

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
        self.missing_value_strategy = 'auto'
        self.handle_outliers = True
        self.outlier_method = 'iqr'
        self.outlier_multiplier = 1.5
        self.outlier_columns = ['age', 'salary']
        self.type_mapping = {}
        self.ranges = {'age': (0, 120), 'salary': (0, 1000000)}
        self.lowercase_text = True
        self.numeric_fill_strategy = 'median'
        self.categorical_fill_strategy = 'mode'
```

## Best Practices
- Always keep original data
- Log all cleaning steps
- Validate data quality
- Handle missing values appropriately
- Remove duplicates early
- Check for outliers
- Validate data types
- Document assumptions

## Checklist
- [ ] Loaded and inspected data
- [ ] Removed duplicates
- [ ] Handled missing values
- [ ] Fixed data types
- [ ] Removed/handled outliers
- [ ] Normalized text fields
- [ ] Validated data quality
- [ ] Generated cleaning report
- [ ] Saved cleaned data
