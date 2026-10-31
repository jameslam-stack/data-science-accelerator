"""
Data Cleaning Pipeline Module

A self-contained, configurable data cleaning module with a DataCleaner class,
clean() function, and CLI interface.

Usage:
    from data_cleaning_pipeline import DataCleaner, clean
    
    cleaner = DataCleaner(df)
    cleaned_df = cleaner.fit_transform()
    
    # Or use the convenience function:
    cleaned_df = clean(df, config={...})
    
    # Or via CLI:
    python data_cleaning_pipeline.py input.csv output.csv
"""

import argparse
import sys
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Callable
import pandas as pd
import numpy as np


@dataclass
class CleaningConfig:
    """Configuration for data cleaning operations."""
    # Whitespace handling
    strip_whitespace: bool = True
    
    # Missing value handling
    missing_value_strategy: str = "mean"  # "mean", "median", "mode", "drop", "fill_constant"
    missing_fill_value: Any = None
    
    # Outlier handling
    handle_outliers: bool = True
    outlier_method: str = "iqr"  # "iqr", "zscore", "cap"
    outlier_multiplier: float = 1.5
    outlier_columns: List[str] = field(default_factory=lambda: ["age", "salary"])
    capping_bounds: Optional[Dict[str, tuple]] = None  # e.g., {"age": (0, 120), "salary": (0, None)}
    
    # Duplicate handling
    drop_duplicates: bool = True
    subset_for_duplicates: Optional[List[str]] = None
    
    # Low variance columns
    drop_low_variance: bool = False
    variance_threshold: float = 0.01
    
    # Type inference
    infer_datetime: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}


class DataCleaner:
    """
    A configurable data cleaner that applies standard cleaning steps.
    
    Example:
        cleaner = DataCleaner(df, config=CleaningConfig(strip_whitespace=True))
        cleaned_df = cleaner.fit_transform()
        report = cleaner.get_report()
    """
    
    def __init__(self, df: pd.DataFrame, config: Optional[CleaningConfig] = None,
                 column_types: Optional[Dict[str, str]] = None):
        """
        Initialize the DataCleaner.
        
        Args:
            df: Input DataFrame to clean
            config: CleaningConfig object with cleaning parameters
            column_types: Optional dict mapping column names to types ("numeric", "categorical", "datetime")
        """
        self.df = df.copy()
        self.config = config or CleaningConfig()
        self.column_types = column_types or {}
        self._report = {
            "original_rows": len(df),
            "original_columns": len(df.columns),
            "steps": [],
            "outliers_found": {},
            "missing_values_handled": {}
        }
    
    def _get_column_type(self, col: str) -> str:
        """Infer column type if not explicitly specified."""
        if col in self.column_types:
            return self.column_types[col]
        
        dtype = self.df[col].dtype
        if pd.api.types.is_numeric_dtype(dtype):
            return "numeric"
        elif self.config.infer_datetime:
            try:
                pd.to_datetime(self.df[col], errors='raise')
                return "datetime"
            except (ValueError, TypeError):
                pass
        return "categorical"
    
    def _step_counter(self, step_name: str) -> Callable:
        """Decorator to track pipeline steps in the report."""
        def decorator(func):
            def wrapper(*args, **kwargs):
                result = func(*args, **kwargs)
                self._report["steps"].append({
                    "step": step_name,
                    "rows_before": len(args[0].df) if args else len(self.df),
                    "rows_after": len(result) if isinstance(result, pd.DataFrame) else len(args[0].df)
                })
                return result
            return wrapper
        return decorator
    
    def clean_whitespace(self) -> 'DataCleaner':
        """Strip leading/trailing whitespace from string columns."""
        if not self.config.strip_whitespace:
            return self
        
        string_cols = self.df.select_dtypes(include=['object', 'string']).columns
        for col in string_cols:
            if self.df[col].apply(lambda x: isinstance(x, str)).any():
                original_values = self.df[col].nunique()
                self.df[col] = self.df[col].astype(str).str.strip()
                new_values = self.df[col].nunique()
                self._report["steps"].append({
                    "step": "strip_whitespace",
                    "column": col,
                    "unique_before": original_values,
                    "unique_after": new_values
                })
        return self
    
    def handle_missing_values(self) -> 'DataCleaner':
        """Handle missing values according to configured strategy."""
        strategy = self.config.missing_value_strategy.lower()
        missing_counts = self.df.isnull().sum()
        cols_with_missing = missing_counts[missing_counts > 0].to_dict()
        
        if not cols_with_missing:
            return self
        
        for col, count in cols_with_missing.items():
            col_type = self._get_column_type(col)
            
            if strategy == "drop":
                self.df = self.df.dropna(subset=[col])
                self._report["missing_values_handled"][col] = {
                    "strategy": "dropped",
                    "count": count
                }
            elif strategy == "mean" or (strategy == "auto" and col_type == "numeric"):
                fill_val = self.df[col].mean()
                self.df[col] = self.df[col].fillna(fill_val)
                self._report["missing_values_handled"][col] = {
                    "strategy": "mean",
                    "value": fill_val,
                    "count": count
                }
            elif strategy == "median" or (strategy == "auto" and col_type != "numeric"):
                fill_val = self.df[col].median() if col_type == "numeric" else self.df[col].mode().iloc[0] if len(self.df[col].mode()) > 0 else ""
                self.df[col] = self.df[col].fillna(fill_val)
                self._report["missing_values_handled"][col] = {
                    "strategy": "median/mode",
                    "value": fill_val,
                    "count": count
                }
            elif strategy == "mode":
                fill_val = self.df[col].mode().iloc[0] if len(self.df[col].mode()) > 0 else ""
                self.df[col] = self.df[col].fillna(fill_val)
                self._report["missing_values_handled"][col] = {
                    "strategy": "mode",
                    "value": fill_val,
                    "count": count
                }
            elif strategy == "fill_constant":
                fill_val = self.config.missing_fill_value if self.config.missing_fill_value is not None else 0
                self.df[col] = self.df[col].fillna(fill_val)
                self._report["missing_values_handled"][col] = {
                    "strategy": "constant",
                    "value": fill_val,
                    "count": count
                }
        
        return self
    
    def detect_outliers_iqr(self, col: str) -> tuple:
        """Detect outliers using IQR method."""
        Q1 = self.df[col].quantile(0.25)
        Q3 = self.df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - self.config.outlier_multiplier * IQR
        upper = Q3 + self.config.outlier_multiplier * IQR
        outliers = self.df[(self.df[col] < lower) | (self.df[col] > upper)]
        return outliers, lower, upper
    
    def detect_outliers_zscore(self, col: str) -> tuple:
        """Detect outliers using Z-score method."""
        mean = self.df[col].mean()
        std = self.df[col].std()
        if std == 0:
            return pd.DataFrame(), mean - 1, mean + 1
        z_scores = np.abs((self.df[col] - mean) / std)
        outliers = self.df[z_scores > self.config.outlier_multiplier]
        return outliers, mean - self.config.outlier_multiplier * std, mean + self.config.outlier_multiplier * std
    
    def handle_outliers(self) -> 'DataCleaner':
        """Handle outliers according to configured method."""
        if not self.config.handle_outliers:
            return self
        
        for col in self.config.outlier_columns:
            if col not in self.df.columns:
                continue
            
            if not pd.api.types.is_numeric_dtype(self.df[col]):
                continue
            
            if self.config.outlier_method == "iqr":
                outliers, lower, upper = self.detect_outliers_iqr(col)
            elif self.config.outlier_method == "zscore":
                outliers, lower, upper = self.detect_outliers_zscore(col)
            else:
                continue
            
            self._report["outliers_found"][col] = {
                "count": len(outliers),
                "method": self.config.outlier_method,
                "lower_bound": lower,
                "upper_bound": upper
            }
            
            if self.config.capping_bounds and col in self.config.capping_bounds:
                lb, ub = self.config.capping_bounds[col]
                if lb is not None:
                    self.df[col] = self.df[col].clip(lower=lb)
                if ub is not None:
                    self.df[col] = self.df[col].clip(upper=ub)
        
        return self
    
    def remove_duplicates(self) -> 'DataCleaner':
        """Remove duplicate rows."""
        if not self.config.drop_duplicates:
            return self
        
        original_len = len(self.df)
        subset = self.config.subset_for_duplicates
        self.df = self.df.drop_duplicates(subset=subset, keep='first')
        rows_removed = original_len - len(self.df)
        
        self._report["steps"].append({
            "step": "remove_duplicates",
            "rows_before": original_len,
            "rows_after": len(self.df),
            "rows_removed": rows_removed
        })
        
        return self
    
    def drop_low_variance_columns(self) -> 'DataCleaner':
        """Drop columns with low variance."""
        if not self.config.drop_low_variance:
            return self
        
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        to_drop = []
        
        for col in numeric_cols:
            var = self.df[col].var()
            if pd.isna(var) or var < self.config.variance_threshold:
                to_drop.append(col)
                self._report["steps"].append({
                    "step": "drop_low_variance",
                    "column": col,
                    "variance": var
                })
        
        if to_drop:
            self.df = self.df.drop(columns=to_drop)
        
        return self
    
    def infer_and_convert_types(self) -> 'DataCleaner':
        """Convert columns to appropriate types."""
        for col in self.df.columns:
            if col in self.column_types:
                target_type = self.column_types[col]
                try:
                    if target_type == "numeric":
                        self.df[col] = pd.to_numeric(self.df[col], errors='coerce')
                    elif target_type == "datetime":
                        self.df[col] = pd.to_datetime(self.df[col])
                except Exception:
                    pass
        
        return self
    
    def get_report(self) -> Dict[str, Any]:
        """Get cleaning report."""
        report = self._report.copy()
        report["final_rows"] = len(self.df)
        report["final_columns"] = len(self.df.columns)
        report["rows_removed"] = report["original_rows"] - report["final_rows"]
        report["columns_removed"] = report["original_columns"] - report["final_columns"]
        return report
    
    def fit_transform(self) -> pd.DataFrame:
        """
        Run the full cleaning pipeline and return cleaned DataFrame.
        
        Steps:
        1. Strip whitespace from string columns
        2. Handle missing values
        3. Handle outliers
        4. Remove duplicates
        5. Convert types
        6. Drop low variance columns
        """
        return (self
                .clean_whitespace()
                .handle_missing_values()
                .handle_outliers()
                .remove_duplicates()
                .infer_and_convert_types()
                .drop_low_variance_columns()
                .df)
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform new data using fitted parameters from training data."""
        original_df = self.df
        self.df = df.copy()
        result = self.fit_transform()
        self.df = original_df
        return result


def clean(df: pd.DataFrame, 
          config: Optional[Dict[str, Any]] = None,
          column_types: Optional[Dict[str, str]] = None) -> pd.DataFrame:
    """
    Convenience function to clean a DataFrame with default or custom configuration.
    
    Args:
        df: Input DataFrame to clean
        config: Optional dictionary of cleaning parameters (will be converted to CleaningConfig)
        column_types: Optional dict mapping column names to types
    
    Returns:
        Cleaned DataFrame
    
    Example:
        cleaned = clean(df, config={
            'missing_value_strategy': 'median',
            'outlier_columns': ['age', 'salary']
        })
    """
    if config is None:
        config_obj = CleaningConfig()
    else:
        config_obj = CleaningConfig(**config)
    
    cleaner = DataCleaner(df, config=config_obj, column_types=column_types)
    return cleaner.fit_transform()


def cli():
    """Command-line interface for data cleaning."""
    parser = argparse.ArgumentParser(
        description="Clean a CSV file with standard data cleaning operations.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s input.csv output.csv
  %(prog)s input.csv output.csv --strategy median --outliers-off
  %(prog)s input.csv output.csv --missing-fill 0 --capping age=0,120;salary=0,None
        """
    )
    
    parser.add_argument("input", help="Input CSV file path")
    parser.add_argument("output", help="Output CSV file path")
    parser.add_argument("--strategy", choices=["mean", "median", "mode", "constant", "drop"],
                        default="mean", help="Missing value imputation strategy")
    parser.add_argument("--missing-fill", type=float, default=None,
                        help="Value to use for constant missing value strategy")
    parser.add_argument("--outliers-off", action="store_true",
                        help="Disable outlier handling")
    parser.add_argument("--outlier-method", choices=["iqr", "zscore"], default="iqr",
                        help="Method for outlier detection")
    parser.add_argument("--no-dedup", action="store_true",
                        help="Disable duplicate removal")
    parser.add_argument("--capping", type=str, default=None,
                        help="Capping bounds as col=lb,ub,... e.g. age=0,120,salary=0,None")
    parser.add_argument("--type-map", type=str, default=None,
                        help="JSON string mapping column names to types (e.g., '{\"age\":\"numeric\"}')")
    parser.add_argument("--report", type=str, default=None,
                        help="Path to save cleaning report as JSON")
    
    args = parser.parse_args()
    
    # Load data
    try:
        df = pd.read_csv(args.input)
    except Exception as e:
        print(f"Error reading input file: {e}", file=sys.stderr)
        sys.exit(1)
    
    # Parse capping bounds (format: col=lb,ub;col=lb,ub;...)
    capping_bounds = None
    if args.capping:
        capping_bounds = {}
        for part in args.capping.split(';'):
            part = part.strip()
            if not part or '=' not in part:
                continue
            col, bounds = part.split('=', 1)
            col = col.strip()
            lb, ub = bounds.split(',')
            lb = float(lb.strip())
            ub = None if ub.strip().lower() == 'none' else float(ub.strip())
            capping_bounds[col] = (lb, ub)

    # Build config
    import json
    config_dict = {
        "missing_value_strategy": args.strategy,
        "handle_outliers": not args.outliers_off,
        "outlier_method": args.outlier_method,
        "drop_duplicates": not args.no_dedup,
        "missing_fill_value": args.missing_fill
    }
    
    if args.type_map:
        try:
            column_types = json.loads(args.type_map)
        except json.JSONDecodeError as e:
            print(f"Error parsing type map: {e}", file=sys.stderr)
            sys.exit(1)
    else:
        column_types = None
    
    # Clean data
    cleaner = DataCleaner(df, config=CleaningConfig(**config_dict), column_types=column_types)
    cleaned_df = cleaner.fit_transform()
    
    # Save output
    try:
        cleaned_df.to_csv(args.output, index=False)
        print(f"Cleaned data saved to {args.output}")
        print(f"Rows: {len(df)} -> {len(cleaned_df)}")
        print(f"Columns: {len(df.columns)} -> {len(cleaned_df.columns)}")
    except Exception as e:
        print(f"Error writing output file: {e}", file=sys.stderr)
        sys.exit(1)
    
    # Save report if requested
    if args.report:
        try:
            with open(args.report, 'w') as f:
                json.dump(cleaner.get_report(), f, indent=2)
            print(f"Cleaning report saved to {args.report}")
        except Exception as e:
            print(f"Error writing report: {e}", file=sys.stderr)


if __name__ == "__main__":
    cli()
