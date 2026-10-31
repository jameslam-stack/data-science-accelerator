import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from IPython.display import display

from sklearn.base import clone
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import (
    GridSearchCV,
    TimeSeriesSplit,
    cross_val_score,
)

from activity import GuidedActivity


# =========================================================
# CONFIGURATION
# =========================================================

RANDOM_STATE = 42
N_SPLITS = 5
HOLDOUT_PERIODS = 18
SEASONAL_PERIOD = 12


# =========================================================
# DATASET
# =========================================================

rng = np.random.default_rng(RANDOM_STATE)

dates = pd.date_range(
    start="2018-01-01",
    periods=96,
    freq="MS",
)

time_index = np.arange(len(dates))

trend = 0.85 * time_index

seasonality = (
    22
    * np.sin(
        2 * np.pi * time_index / SEASONAL_PERIOD
    )
)

promotion = (
    (time_index % 6 == 0)
    | (time_index % 6 == 1)
).astype(int)

promotion_effect = 14 * promotion

noise = rng.normal(
    loc=0,
    scale=6,
    size=len(dates),
)

sales = (
    120
    + trend
    + seasonality
    + promotion_effect
    + noise
)

time_series_data = pd.DataFrame(
    {
        "date": dates,
        "sales": sales.round(2),
        "promotion": promotion,
    }
)


# =========================================================
# PREVIEW
# =========================================================

def show_dataset(namespace):
    data = namespace["time_series_data"]

    print("Dataset available as: time_series_data")
    print("Date column: date")
    print("Target column: sales")
    print("Frequency: monthly")
    print("Dataset shape:", data.shape)
    print("Seasonal period:", namespace["SEASONAL_PERIOD"])
    print("Holdout periods:", namespace["HOLDOUT_PERIODS"])

    display(data.head())


# =========================================================
# VALIDATORS
# =========================================================

def validate_data_copy(namespace):
    if "data" not in namespace:
        return False

    result = namespace["data"]
    expected = namespace["time_series_data"]

    return (
        isinstance(result, pd.DataFrame)
        and result.equals(expected)
        and result is not expected
    )


def validate_sorted_data(namespace):
    if "data" not in namespace:
        return False

    result = namespace["data"]

    if not isinstance(result, pd.DataFrame):
        return False

    expected = (
        namespace["time_series_data"]
        .copy()
        .sort_values("date")
        .reset_index(drop=True)
    )

    return (
        result.equals(expected)
        and result["date"].is_monotonic_increasing
    )


def validate_frequency(namespace):
    if "inferred_frequency" not in namespace:
        return False

    return namespace["inferred_frequency"] in {
        "MS",
        "ME",
        "M",
    }


def validate_duplicate_dates(namespace):
    if "duplicate_dates" not in namespace:
        return False

    expected = int(
        namespace["data"]["date"]
        .duplicated()
        .sum()
    )

    result = namespace["duplicate_dates"]

    return (
        isinstance(result, (int, np.integer))
        and int(result) == expected
    )


def validate_missing_summary(namespace):
    if "missing_summary" not in namespace:
        return False

    result = namespace["missing_summary"]

    expected = (
        namespace["data"]
        .isna()
        .sum()
        .sort_values(ascending=False)
    )

    return (
        isinstance(result, pd.Series)
        and result.equals(expected)
    )


def validate_target_summary(namespace):
    if "target_summary" not in namespace:
        return False

    result = namespace["target_summary"]

    if not isinstance(result, pd.Series):
        return False

    required = {
        "mean",
        "median",
        "standard_deviation",
        "minimum",
        "maximum",
    }

    return (
        set(result.index) == required
        and np.isclose(
            result["mean"],
            namespace["data"]["sales"].mean(),
        )
    )


def validate_calendar_features(namespace):
    data = namespace["data"]

    required = {
        "month",
        "month_sin",
        "month_cos",
    }

    if not required.issubset(data.columns):
        return False

    expected_month = data["date"].dt.month

    expected_sin = np.sin(
        2 * np.pi * expected_month / 12
    )

    expected_cos = np.cos(
        2 * np.pi * expected_month / 12
    )

    return (
        data["month"].equals(expected_month)
        and np.allclose(
            data["month_sin"],
            expected_sin,
        )
        and np.allclose(
            data["month_cos"],
            expected_cos,
        )
    )


def validate_lag_features(namespace):
    data = namespace["data"]

    required = {
        "lag_1",
        "lag_2",
        "lag_3",
        "lag_12",
    }

    if not required.issubset(data.columns):
        return False

    sales_series = data["sales"]

    return (
        data["lag_1"].equals(
            sales_series.shift(1)
        )
        and data["lag_2"].equals(
            sales_series.shift(2)
        )
        and data["lag_3"].equals(
            sales_series.shift(3)
        )
        and data["lag_12"].equals(
            sales_series.shift(
                namespace["SEASONAL_PERIOD"]
            )
        )
    )


def validate_rolling_features(namespace):
    data = namespace["data"]

    required = {
        "rolling_mean_3",
        "rolling_mean_6",
    }

    if not required.issubset(data.columns):
        return False

    shifted_sales = data["sales"].shift(1)

    expected_3 = (
        shifted_sales
        .rolling(window=3)
        .mean()
    )

    expected_6 = (
        shifted_sales
        .rolling(window=6)
        .mean()
    )

    return (
        data["rolling_mean_3"].equals(
            expected_3
        )
        and data["rolling_mean_6"].equals(
            expected_6
        )
    )


def validate_modelling_data(namespace):
    if "modelling_data" not in namespace:
        return False

    result = namespace["modelling_data"]

    expected = (
        namespace["data"]
        .dropna()
        .reset_index(drop=True)
    )

    return (
        isinstance(result, pd.DataFrame)
        and result.equals(expected)
        and not result.isna().any().any()
    )


def validate_feature_columns(namespace):
    if "feature_columns" not in namespace:
        return False

    expected = [
        "promotion",
        "month_sin",
        "month_cos",
        "lag_1",
        "lag_2",
        "lag_3",
        "lag_12",
        "rolling_mean_3",
        "rolling_mean_6",
    ]

    return (
        isinstance(
            namespace["feature_columns"],
            list,
        )
        and namespace["feature_columns"] == expected
    )


def validate_chronological_split(namespace):
    required = {
        "train_data",
        "test_data",
        "X_train",
        "X_test",
        "y_train",
        "y_test",
    }

    if not required.issubset(namespace):
        return False

    modelling_data = namespace["modelling_data"]
    holdout = namespace["HOLDOUT_PERIODS"]

    expected_train = (
        modelling_data
        .iloc[:-holdout]
        .copy()
    )

    expected_test = (
        modelling_data
        .iloc[-holdout:]
        .copy()
    )

    features = namespace["feature_columns"]

    return (
        namespace["train_data"].equals(
            expected_train
        )
        and namespace["test_data"].equals(
            expected_test
        )
        and namespace["X_train"].equals(
            expected_train[features]
        )
        and namespace["X_test"].equals(
            expected_test[features]
        )
        and namespace["y_train"].equals(
            expected_train["sales"]
        )
        and namespace["y_test"].equals(
            expected_test["sales"]
        )
    )


def validate_split_order(namespace):
    if "split_is_chronological" not in namespace:
        return False

    result = namespace[
        "split_is_chronological"
    ]

    expected = (
        namespace["train_data"]["date"].max()
        < namespace["test_data"]["date"].min()
    )

    return (
        isinstance(result, (bool, np.bool_))
        and bool(result) == bool(expected)
        and bool(result) is True
    )


def validate_baseline_predictions(namespace):
    if "baseline_predictions" not in namespace:
        return False

    result = namespace[
        "baseline_predictions"
    ]

    expected = (
        namespace["test_data"]["lag_12"]
        .to_numpy()
    )

    return (
        isinstance(result, np.ndarray)
        and np.allclose(
            result,
            expected,
        )
    )


def validate_baseline_metrics(namespace):
    if "baseline_metrics" not in namespace:
        return False

    result = namespace[
        "baseline_metrics"
    ]

    if not isinstance(result, pd.Series):
        return False

    required = {
        "mae",
        "rmse",
    }

    if set(result.index) != required:
        return False

    y_test = namespace["y_test"]

    predictions = namespace[
        "baseline_predictions"
    ]

    expected_mae = mean_absolute_error(
        y_test,
        predictions,
    )

    expected_rmse = (
        mean_squared_error(
            y_test,
            predictions,
        )
        ** 0.5
    )

    return (
        np.isclose(
            result["mae"],
            expected_mae,
        )
        and np.isclose(
            result["rmse"],
            expected_rmse,
        )
    )


def validate_cv(namespace):
    if "cv" not in namespace:
        return False

    result = namespace["cv"]

    return (
        isinstance(
            result,
            TimeSeriesSplit,
        )
        and result.n_splits
        == namespace["N_SPLITS"]
    )


def validate_model(namespace):
    if "model" not in namespace:
        return False

    result = namespace["model"]

    return (
        isinstance(
            result,
            RandomForestRegressor,
        )
        and result.n_estimators == 200
        and result.random_state
        == namespace["RANDOM_STATE"]
        and result.n_jobs == -1
    )


def validate_cv_scores(namespace):
    if "model_cv_mae" not in namespace:
        return False

    result = namespace[
        "model_cv_mae"
    ]

    return (
        isinstance(result, np.ndarray)
        and len(result)
        == namespace["N_SPLITS"]
        and np.all(result >= 0)
    )


def validate_validation_summary(namespace):
    if "validation_summary" not in namespace:
        return False

    result = namespace[
        "validation_summary"
    ]

    if not isinstance(result, pd.Series):
        return False

    return (
        set(result.index)
        == {
            "mean_cv_mae",
            "std_cv_mae",
        }
        and np.isclose(
            result["mean_cv_mae"],
            namespace["model_cv_mae"].mean(),
        )
    )


def validate_parameter_grid(namespace):
    if "parameter_grid" not in namespace:
        return False

    result = namespace[
        "parameter_grid"
    ]

    if not isinstance(result, dict):
        return False

    return (
        list(
            result["n_estimators"]
        )
        == [100, 200]
        and list(
            result["max_depth"]
        )
        == [3, 6, None]
        and list(
            result["min_samples_leaf"]
        )
        == [1, 2, 4]
    )


def validate_grid_search(namespace):
    if "grid_search" not in namespace:
        return False

    result = namespace[
        "grid_search"
    ]

    return (
        isinstance(
            result,
            GridSearchCV,
        )
        and result.scoring
        == "neg_mean_absolute_error"
        and result.cv is namespace["cv"]
        and result.refit is True
    )


def validate_fitted_grid(namespace):
    if "grid_search" not in namespace:
        return False

    result = namespace["grid_search"]

    return (
        hasattr(result, "best_estimator_")
        and hasattr(result, "best_params_")
        and hasattr(result, "best_score_")
        and hasattr(result, "cv_results_")
    )


def validate_tuning_results(namespace):
    if "tuning_results" not in namespace:
        return False

    result = namespace[
        "tuning_results"
    ]

    if not isinstance(result, pd.DataFrame):
        return False

    required = {
        "rank_test_score",
        "mean_test_score",
        "std_test_score",
        "params",
    }

    return (
        required.issubset(
            result.columns
        )
        and len(result) > 0
        and result.iloc[0][
            "rank_test_score"
        ] == 1
    )


def validate_best_model(namespace):
    if "best_model" not in namespace:
        return False

    return (
        isinstance(
            namespace["best_model"],
            RandomForestRegressor,
        )
        and namespace["best_model"]
        is namespace[
            "grid_search"
        ].best_estimator_
    )


def validate_final_model(namespace):
    if "final_model" not in namespace:
        return False

    result = namespace["final_model"]

    return (
        isinstance(
            result,
            RandomForestRegressor,
        )
        and result
        is not namespace["best_model"]
        and hasattr(
            result,
            "feature_importances_",
        )
    )


def validate_test_predictions(namespace):
    if "test_predictions" not in namespace:
        return False

    result = namespace[
        "test_predictions"
    ]

    return (
        isinstance(result, np.ndarray)
        and result.ndim == 1
        and len(result)
        == len(namespace["y_test"])
    )


def validate_holdout_metrics(namespace):
    if "holdout_metrics" not in namespace:
        return False

    result = namespace[
        "holdout_metrics"
    ]

    if not isinstance(result, pd.Series):
        return False

    required = {
        "mae",
        "rmse",
        "mape_percent",
        "r2",
    }

    return (
        set(result.index) == required
        and result["mae"] >= 0
        and result["rmse"] >= 0
        and result["mape_percent"] >= 0
        and np.isfinite(result["r2"])
    )


def validate_forecast_results(namespace):
    if "forecast_results" not in namespace:
        return False

    result = namespace[
        "forecast_results"
    ]

    required = {
        "date",
        "actual",
        "predicted",
        "residual",
        "absolute_error",
    }

    return (
        isinstance(result, pd.DataFrame)
        and required.issubset(
            result.columns
        )
        and len(result)
        == len(namespace["y_test"])
    )


def validate_residual_summary(namespace):
    if "residual_summary" not in namespace:
        return False

    result = namespace[
        "residual_summary"
    ]

    if not isinstance(result, pd.Series):
        return False

    required = {
        "mean_residual",
        "median_residual",
        "residual_std",
        "largest_absolute_error",
    }

    return (
        set(result.index) == required
        and np.isfinite(
            result.astype(float)
        ).all()
    )


def validate_largest_errors(namespace):
    if "largest_errors" not in namespace:
        return False

    result = namespace[
        "largest_errors"
    ]

    return (
        isinstance(result, pd.DataFrame)
        and len(result) == 10
        and result[
            "absolute_error"
        ].is_monotonic_decreasing
    )


def validate_feature_importance(namespace):
    if "feature_importance" not in namespace:
        return False

    result = namespace[
        "feature_importance"
    ]

    required = {
        "feature",
        "importance",
    }

    return (
        isinstance(result, pd.DataFrame)
        and required.issubset(
            result.columns
        )
        and len(result)
        == len(
            namespace["feature_columns"]
        )
        and result[
            "importance"
        ].is_monotonic_decreasing
    )


def validate_metadata(namespace):
    if "model_metadata" not in namespace:
        return False

    result = namespace[
        "model_metadata"
    ]

    if not isinstance(result, dict):
        return False

    required = {
        "model_type",
        "created_at_utc",
        "frequency",
        "seasonal_period",
        "holdout_periods",
        "cross_validation_folds",
        "primary_metric",
        "best_parameters",
        "cross_validation_mae",
        "holdout_metrics",
        "feature_columns",
        "target_column",
        "date_column",
    }

    return (
        required.issubset(result)
        and result["model_type"]
        == "RandomForestRegressor"
        and result["primary_metric"]
        == "MAE"
    )


def validate_model_bundle(namespace):
    required = {
        "model_bundle",
        "artifact_path",
    }

    if not required.issubset(namespace):
        return False

    bundle = namespace[
        "model_bundle"
    ]

    path = Path(
        namespace["artifact_path"]
    )

    if not path.exists():
        return False

    try:
        loaded = joblib.load(path)
    except Exception:
        return False

    return (
        isinstance(bundle, dict)
        and isinstance(loaded, dict)
        and isinstance(
            loaded.get("model"),
            RandomForestRegressor,
        )
    )


def validate_forecast_function(namespace):
    if "forecast_next_period" not in namespace:
        return False

    function = namespace[
        "forecast_next_period"
    ]

    if not callable(function):
        return False

    try:
        result = function(
            namespace["model_bundle"],
            namespace["time_series_data"],
            promotion=0,
        )
    except Exception:
        return False

    return (
        isinstance(result, pd.DataFrame)
        and {
            "date",
            "prediction",
        }.issubset(result.columns)
        and len(result) == 1
    )


# =========================================================
# ACTIVITY STEPS
# =========================================================

steps = [

    {
        "group": "1. Time-series understanding & quality",
        "title": "Step 1: Create a working copy",
        "instruction": (
            "Create an independent DataFrame called data by "
            "copying time_series_data."
        ),
        "placeholder": (
            "data = your_source_dataframe.copy()"
        ),
        "validator": validate_data_copy,
    },

    {
        "group": "1. Time-series understanding & quality",
        "title": "Step 2: Sort observations chronologically",
        "instruction": (
            "Sort data by date from earliest to latest and reset "
            "the index."
        ),
        "placeholder": (
            "data = (\n"
            "    data\n"
            '    .sort_values("date")\n'
            "    .reset_index(drop=True)\n"
            ")"
        ),
        "validator": validate_sorted_data,
    },

    {
        "group": "1. Time-series understanding & quality",
        "title": "Step 3: Confirm the time frequency",
        "instruction": (
            "Use pd.infer_freq on the date column and store the "
            "result in inferred_frequency."
        ),
        "placeholder": (
            "inferred_frequency = pd.infer_freq(\n"
            '    data["date"]\n'
            ")\n\n"
            "print(inferred_frequency)"
        ),
        "validator": validate_frequency,
    },

    {
        "group": "1. Time-series understanding & quality",
        "title": "Step 4: Check for duplicate timestamps",
        "instruction": (
            "Create duplicate_dates containing the number of "
            "duplicated dates."
        ),
        "placeholder": (
            "duplicate_dates = int(\n"
            '    data["date"]\n'
            "    .duplicated()\n"
            "    .sum()\n"
            ")\n\n"
            "print(duplicate_dates)"
        ),
        "validator": validate_duplicate_dates,
    },

    {
        "group": "1. Time-series understanding & quality",
        "title": "Step 5: Audit missing values",
        "instruction": (
            "Create missing_summary containing missing values per "
            "column, sorted from highest to lowest."
        ),
        "placeholder": (
            "missing_summary = (\n"
            "    data\n"
            "    .isna()\n"
            "    .sum()\n"
            "    .sort_values(ascending=False)\n"
            ")\n\n"
            "print(missing_summary)"
        ),
        "validator": validate_missing_summary,
    },

    {
        "group": "1. Time-series understanding & quality",
        "title": "Step 6: Summarise the target series",
        "instruction": (
            "Create target_summary containing mean, median, "
            "standard deviation, minimum and maximum sales."
        ),
        "placeholder": (
            "target_summary = pd.Series(\n"
            "    {\n"
            '        "mean": data["sales"].mean(),\n'
            '        "median": data["sales"].median(),\n'
            '        "standard_deviation": data["sales"].std(),\n'
            '        "minimum": data["sales"].min(),\n'
            '        "maximum": data["sales"].max(),\n'
            "    }\n"
            ")"
        ),
        "validator": validate_target_summary,
    },

    {
        "group": "2. Time-series feature engineering",
        "title": "Step 7: Create calendar features",
        "instruction": (
            "Create month, month_sin and month_cos to represent "
            "annual seasonality."
        ),
        "placeholder": (
            'data["month"] = data["date"].dt.month\n\n'
            'data["month_sin"] = np.sin(\n'
            '    2 * np.pi * data["month"] / 12\n'
            ")\n\n"
            'data["month_cos"] = np.cos(\n'
            '    2 * np.pi * data["month"] / 12\n'
            ")"
        ),
        "validator": validate_calendar_features,
    },

    {
        "group": "2. Time-series feature engineering",
        "title": "Step 8: Create lag features",
        "instruction": (
            "Create lag_1, lag_2, lag_3 and lag_12 from sales."
        ),
        "placeholder": (
            'data["lag_1"] = data["sales"].shift(1)\n'
            'data["lag_2"] = data["sales"].shift(2)\n'
            'data["lag_3"] = data["sales"].shift(3)\n'
            'data["lag_12"] = data["sales"].shift(\n'
            "    SEASONAL_PERIOD\n"
            ")"
        ),
        "validator": validate_lag_features,
    },

    {
        "group": "2. Time-series feature engineering",
        "title": "Step 9: Create leakage-safe rolling features",
        "instruction": (
            "Create rolling_mean_3 and rolling_mean_6 using sales "
            "shifted by one period before calculating each rolling "
            "mean."
        ),
        "placeholder": (
            "shifted_sales = (\n"
            '    data["sales"].shift(1)\n'
            ")\n\n"
            'data["rolling_mean_3"] = (\n'
            "    shifted_sales\n"
            "    .rolling(window=3)\n"
            "    .mean()\n"
            ")\n\n"
            'data["rolling_mean_6"] = (\n'
            "    shifted_sales\n"
            "    .rolling(window=6)\n"
            "    .mean()\n"
            ")"
        ),
        "validator": validate_rolling_features,
    },

    {
        "group": "2. Time-series feature engineering",
        "title": "Step 10: Create the modelling dataset",
        "instruction": (
            "Create modelling_data by dropping rows containing "
            "missing lag or rolling values and resetting the index."
        ),
        "placeholder": (
            "modelling_data = (\n"
            "    data\n"
            "    .dropna()\n"
            "    .reset_index(drop=True)\n"
            ")"
        ),
        "validator": validate_modelling_data,
    },

    {
        "group": "2. Time-series feature engineering",
        "title": "Step 11: Define modelling features",
        "instruction": (
            "Create feature_columns containing the exogenous, "
            "calendar, lag and rolling features."
        ),
        "placeholder": (
            "feature_columns = [\n"
            '    "promotion",\n'
            '    "month_sin",\n'
            '    "month_cos",\n'
            '    "lag_1",\n'
            '    "lag_2",\n'
            '    "lag_3",\n'
            '    "lag_12",\n'
            '    "rolling_mean_3",\n'
            '    "rolling_mean_6",\n'
            "]"
        ),
        "validator": validate_feature_columns,
    },

    {
        "group": "3. Chronological holdout strategy",
        "title": "Step 12: Create a chronological train-test split",
        "instruction": (
            "Reserve the final HOLDOUT_PERIODS observations for "
            "testing and keep all earlier observations for training."
        ),
        "placeholder": (
            "train_data = (\n"
            "    modelling_data\n"
            "    .iloc[:-HOLDOUT_PERIODS]\n"
            "    .copy()\n"
            ")\n\n"
            "test_data = (\n"
            "    modelling_data\n"
            "    .iloc[-HOLDOUT_PERIODS:]\n"
            "    .copy()\n"
            ")\n\n"
            "X_train = train_data[feature_columns]\n"
            "X_test = test_data[feature_columns]\n\n"
            'y_train = train_data["sales"]\n'
            'y_test = test_data["sales"]'
        ),
        "validator": validate_chronological_split,
    },

    {
        "group": "3. Chronological holdout strategy",
        "title": "Step 13: Verify temporal order",
        "instruction": (
            "Create split_is_chronological confirming that the "
            "training period ends before the test period begins."
        ),
        "placeholder": (
            "split_is_chronological = (\n"
            '    train_data["date"].max()\n'
            '    < test_data["date"].min()\n'
            ")"
        ),
        "validator": validate_split_order,
    },

    {
        "group": "4. Seasonal baseline",
        "title": "Step 14: Create a seasonal-naive forecast",
        "instruction": (
            "Create baseline_predictions using the lag_12 values "
            "from test_data."
        ),
        "placeholder": (
            "baseline_predictions = (\n"
            '    test_data["lag_12"]\n'
            "    .to_numpy()\n"
            ")"
        ),
        "validator": validate_baseline_predictions,
    },

    {
        "group": "4. Seasonal baseline",
        "title": "Step 15: Evaluate the baseline",
        "instruction": (
            "Create baseline_metrics containing MAE and RMSE."
        ),
        "placeholder": (
            "baseline_metrics = pd.Series(\n"
            "    {\n"
            '        "mae": mean_absolute_error(\n'
            "            y_test,\n"
            "            baseline_predictions,\n"
            "        ),\n"
            '        "rmse": (\n'
            "            mean_squared_error(\n"
            "                y_test,\n"
            "                baseline_predictions,\n"
            "            ) ** 0.5\n"
            "        ),\n"
            "    }\n"
            ")"
        ),
        "validator": validate_baseline_metrics,
    },

    {
        "group": "5. Time-aware model validation",
        "title": "Step 16: Configure time-series cross-validation",
        "instruction": (
            "Create cv using TimeSeriesSplit with N_SPLITS."
        ),
        "placeholder": (
            "cv = TimeSeriesSplit(\n"
            "    n_splits=N_SPLITS\n"
            ")"
        ),
        "validator": validate_cv,
    },

    {
        "group": "5. Time-aware model validation",
        "title": "Step 17: Create the forecasting model",
        "instruction": (
            "Create model using RandomForestRegressor with 200 "
            "trees, RANDOM_STATE and n_jobs=-1."
        ),
        "placeholder": (
            "model = RandomForestRegressor(\n"
            "    n_estimators=200,\n"
            "    random_state=RANDOM_STATE,\n"
            "    n_jobs=-1,\n"
            ")"
        ),
        "validator": validate_model,
    },

    {
        "group": "5. Time-aware model validation",
        "title": "Step 18: Cross-validate the model",
        "instruction": (
            "Evaluate model using TimeSeriesSplit and MAE. Store "
            "positive MAE fold scores in model_cv_mae."
        ),
        "placeholder": (
            "model_cv_mae = -cross_val_score(\n"
            "    model,\n"
            "    X_train,\n"
            "    y_train,\n"
            "    cv=cv,\n"
            '    scoring="neg_mean_absolute_error",\n'
            "    n_jobs=-1,\n"
            ")"
        ),
        "validator": validate_cv_scores,
    },

    {
        "group": "5. Time-aware model validation",
        "title": "Step 19: Summarise validation performance",
        "instruction": (
            "Create validation_summary containing mean_cv_mae and "
            "std_cv_mae."
        ),
        "placeholder": (
            "validation_summary = pd.Series(\n"
            "    {\n"
            '        "mean_cv_mae": model_cv_mae.mean(),\n'
            '        "std_cv_mae": model_cv_mae.std(),\n'
            "    }\n"
            ")"
        ),
        "validator": validate_validation_summary,
    },

    {
        "group": "6. Hyperparameter tuning",
        "title": "Step 20: Define a parameter search space",
        "instruction": (
            "Create parameter_grid for n_estimators, max_depth and "
            "min_samples_leaf."
        ),
        "placeholder": (
            "parameter_grid = {\n"
            '    "n_estimators": [100, 200],\n'
            '    "max_depth": [3, 6, None],\n'
            '    "min_samples_leaf": [1, 2, 4],\n'
            "}"
        ),
        "validator": validate_parameter_grid,
    },

    {
        "group": "6. Hyperparameter tuning",
        "title": "Step 21: Configure time-aware grid search",
        "instruction": (
            "Create grid_search using GridSearchCV, cv=cv and "
            "negative MAE as the scoring metric."
        ),
        "placeholder": (
            "grid_search = GridSearchCV(\n"
            "    estimator=model,\n"
            "    param_grid=parameter_grid,\n"
            "    cv=cv,\n"
            '    scoring="neg_mean_absolute_error",\n'
            "    refit=True,\n"
            "    n_jobs=-1,\n"
            ")"
        ),
        "validator": validate_grid_search,
    },

    {
        "group": "6. Hyperparameter tuning",
        "title": "Step 22: Fit the grid search",
        "instruction": (
            "Fit grid_search using training data only."
        ),
        "placeholder": (
            "grid_search.fit(\n"
            "    X_train,\n"
            "    y_train,\n"
            ")"
        ),
        "validator": validate_fitted_grid,
    },

    {
        "group": "6. Hyperparameter tuning",
        "title": "Step 23: Review tuning results",
        "instruction": (
            "Create tuning_results from grid_search.cv_results_ "
            "and sort the best result first."
        ),
        "placeholder": (
            "tuning_results = (\n"
            "    pd.DataFrame(\n"
            "        grid_search.cv_results_\n"
            "    )\n"
            "    [[\n"
            '        "rank_test_score",\n'
            '        "mean_test_score",\n'
            '        "std_test_score",\n'
            '        "params",\n'
            "    ]]\n"
            '    .sort_values("rank_test_score")\n'
            "    .reset_index(drop=True)\n"
            ")"
        ),
        "validator": validate_tuning_results,
    },

    {
        "group": "6. Hyperparameter tuning",
        "title": "Step 24: Retrieve the selected model",
        "instruction": (
            "Store grid_search.best_estimator_ as best_model."
        ),
        "placeholder": (
            "best_model = grid_search.best_estimator_"
        ),
        "validator": validate_best_model,
    },

    {
        "group": "7. Final holdout evaluation",
        "title": "Step 25: Refit the final model",
        "instruction": (
            "Clone best_model as final_model and fit it on all "
            "training observations."
        ),
        "placeholder": (
            "final_model = clone(best_model)\n\n"
            "final_model.fit(\n"
            "    X_train,\n"
            "    y_train,\n"
            ")"
        ),
        "validator": validate_final_model,
    },

    {
        "group": "7. Final holdout evaluation",
        "title": "Step 26: Forecast the holdout period",
        "instruction": (
            "Use final_model to predict X_test and store the "
            "results in test_predictions."
        ),
        "placeholder": (
            "test_predictions = (\n"
            "    final_model.predict(X_test)\n"
            ")"
        ),
        "validator": validate_test_predictions,
    },

    {
        "group": "7. Final holdout evaluation",
        "title": "Step 27: Calculate holdout metrics",
        "instruction": (
            "Create holdout_metrics containing MAE, RMSE, MAPE "
            "percentage and R²."
        ),
        "placeholder": (
            "actual = y_test.to_numpy()\n\n"
            "holdout_metrics = pd.Series(\n"
            "    {\n"
            '        "mae": mean_absolute_error(\n'
            "            actual,\n"
            "            test_predictions,\n"
            "        ),\n"
            '        "rmse": (\n'
            "            mean_squared_error(\n"
            "                actual,\n"
            "                test_predictions,\n"
            "            ) ** 0.5\n"
            "        ),\n"
            '        "mape_percent": float(\n'
            "            np.mean(\n"
            "                np.abs(\n"
            "                    (\n"
            "                        actual\n"
            "                        - test_predictions\n"
            "                    )\n"
            "                    / actual\n"
            "                )\n"
            "            ) * 100\n"
            "        ),\n"
            '        "r2": r2_score(\n'
            "            actual,\n"
            "            test_predictions,\n"
            "        ),\n"
            "    }\n"
            ")"
        ),
        "validator": validate_holdout_metrics,
    },

    {
        "group": "8. Forecast error analysis",
        "title": "Step 28: Create a forecast results table",
        "instruction": (
            "Create forecast_results containing date, actual, "
            "predicted, residual and absolute_error."
        ),
        "placeholder": (
            "forecast_results = pd.DataFrame(\n"
            "    {\n"
            '        "date": test_data["date"].to_numpy(),\n'
            '        "actual": y_test.to_numpy(),\n'
            '        "predicted": test_predictions,\n'
            "    }\n"
            ")\n\n"
            'forecast_results["residual"] = (\n'
            '    forecast_results["actual"]\n'
            '    - forecast_results["predicted"]\n'
            ")\n\n"
            'forecast_results["absolute_error"] = (\n'
            '    forecast_results["residual"].abs()\n'
            ")"
        ),
        "validator": validate_forecast_results,
    },

    {
        "group": "8. Forecast error analysis",
        "title": "Step 29: Summarise residual behaviour",
        "instruction": (
            "Create residual_summary containing mean residual, "
            "median residual, residual standard deviation and "
            "largest absolute error."
        ),
        "placeholder": (
            "residual_summary = pd.Series(\n"
            "    {\n"
            '        "mean_residual": (\n'
            '            forecast_results["residual"].mean()\n'
            "        ),\n"
            '        "median_residual": (\n'
            '            forecast_results["residual"].median()\n'
            "        ),\n"
            '        "residual_std": (\n'
            '            forecast_results["residual"].std()\n'
            "        ),\n"
            '        "largest_absolute_error": (\n'
            '            forecast_results["absolute_error"].max()\n'
            "        ),\n"
            "    }\n"
            ")"
        ),
        "validator": validate_residual_summary,
    },

    {
        "group": "8. Forecast error analysis",
        "title": "Step 30: Inspect the largest errors",
        "instruction": (
            "Create largest_errors containing the 10 observations "
            "with the greatest absolute forecast error."
        ),
        "placeholder": (
            "largest_errors = (\n"
            "    forecast_results\n"
            "    .sort_values(\n"
            '        "absolute_error",\n'
            "        ascending=False,\n"
            "    )\n"
            "    .head(10)\n"
            ")"
        ),
        "validator": validate_largest_errors,
    },

    {
        "group": "9. Model interpretation",
        "title": "Step 31: Inspect feature importance",
        "instruction": (
            "Create feature_importance containing each feature and "
            "its fitted Random Forest importance, sorted highest "
            "to lowest."
        ),
        "placeholder": (
            "feature_importance = (\n"
            "    pd.DataFrame(\n"
            "        {\n"
            '            "feature": feature_columns,\n'
            '            "importance": (\n'
            "                final_model.feature_importances_\n"
            "            ),\n"
            "        }\n"
            "    )\n"
            "    .sort_values(\n"
            '        "importance",\n'
            "        ascending=False,\n"
            "    )\n"
            "    .reset_index(drop=True)\n"
            ")"
        ),
        "validator": validate_feature_importance,
    },

    {
        "group": "10. Packaging & inference",
        "title": "Step 32: Create model metadata",
        "instruction": (
            "Create model_metadata documenting the forecasting "
            "workflow, configuration, tuning results and metrics."
        ),
        "placeholder": (
            "model_metadata = {\n"
            '    "model_type": "RandomForestRegressor",\n'
            '    "created_at_utc": (\n'
            "        datetime.now(timezone.utc)\n"
            "        .isoformat()\n"
            "    ),\n"
            '    "frequency": inferred_frequency,\n'
            '    "seasonal_period": SEASONAL_PERIOD,\n'
            '    "holdout_periods": HOLDOUT_PERIODS,\n'
            '    "cross_validation_folds": N_SPLITS,\n'
            '    "primary_metric": "MAE",\n'
            '    "best_parameters": grid_search.best_params_,\n'
            '    "cross_validation_mae": float(\n'
            "        -grid_search.best_score_\n"
            "    ),\n"
            '    "holdout_metrics": holdout_metrics.to_dict(),\n'
            '    "feature_columns": feature_columns,\n'
            '    "target_column": "sales",\n'
            '    "date_column": "date",\n'
            "}"
        ),
        "validator": validate_metadata,
    },

    {
        "group": "10. Packaging & inference",
        "title": "Step 33: Package and save the model",
        "instruction": (
            "Create model_bundle containing final_model and "
            "model_metadata. Save it using joblib."
        ),
        "placeholder": (
            "model_bundle = {\n"
            '    "model": final_model,\n'
            '    "metadata": model_metadata,\n'
            "}\n\n"
            "artifact_path = Path(\n"
            '    "monthly_sales_forecast.joblib"\n'
            ")\n\n"
            "joblib.dump(\n"
            "    model_bundle,\n"
            "    artifact_path,\n"
            ")"
        ),
        "validator": validate_model_bundle,
    },

    {
        "group": "10. Packaging & inference",
        "title": "Step 34: Create a reusable next-period forecast function",
        "instruction": (
            "Define forecast_next_period(bundle, history, promotion) "
            "to construct the next month's features and return a "
            "one-row forecast."
        ),
        "placeholder": (
            "def forecast_next_period(\n"
            "    bundle,\n"
            "    history,\n"
            "    promotion=0,\n"
            "):\n"
            "    model = bundle[\"model\"]\n"
            "    metadata = bundle[\"metadata\"]\n\n"
            "    history = (\n"
            "        history\n"
            '        .sort_values("date")\n'
            "        .reset_index(drop=True)\n"
            "    )\n\n"
            "    next_date = (\n"
            '        history["date"].max()\n'
            "        + pd.offsets.MonthBegin(1)\n"
            "    )\n\n"
            "    month = next_date.month\n\n"
            "    feature_row = pd.DataFrame(\n"
            "        {\n"
            '            "promotion": [promotion],\n'
            '            "month_sin": [\n'
            "                np.sin(\n"
            "                    2 * np.pi * month / 12\n"
            "                )\n"
            "            ],\n"
            '            "month_cos": [\n'
            "                np.cos(\n"
            "                    2 * np.pi * month / 12\n"
            "                )\n"
            "            ],\n"
            '            "lag_1": [\n'
            '                history["sales"].iloc[-1]\n'
            "            ],\n"
            '            "lag_2": [\n'
            '                history["sales"].iloc[-2]\n'
            "            ],\n"
            '            "lag_3": [\n'
            '                history["sales"].iloc[-3]\n'
            "            ],\n"
            '            "lag_12": [\n'
            '                history["sales"].iloc[-12]\n'
            "            ],\n"
            '            "rolling_mean_3": [\n'
            '                history["sales"]\n'
            "                .iloc[-3:]\n"
            "                .mean()\n"
            "            ],\n"
            '            "rolling_mean_6": [\n'
            '                history["sales"]\n'
            "                .iloc[-6:]\n"
            "                .mean()\n"
            "            ],\n"
            "        }\n"
            "    )\n\n"
            "    feature_row = feature_row[\n"
            '        metadata["feature_columns"]\n'
            "    ]\n\n"
            "    prediction = model.predict(\n"
            "        feature_row\n"
            "    )[0]\n\n"
            "    return pd.DataFrame(\n"
            "        {\n"
            '            "date": [next_date],\n'
            '            "prediction": [prediction],\n'
            "        }\n"
            "    )"
        ),
        "validator": validate_forecast_function,
        "input_height": "760px",
    },
]


# =========================================================
# CREATE ACTIVITY
# =========================================================

def create_activity():
    return GuidedActivity(
        title="Professional time-series forecasting workflow",
        introduction=(
            "Develop a reproducible end-to-end forecasting "
            "workflow using monthly sales data. The activity "
            "covers temporal data-quality checks, chronological "
            "splitting, seasonal and lag features, leakage-safe "
            "rolling statistics, seasonal-naive benchmarking, "
            "TimeSeriesSplit validation, hyperparameter tuning, "
            "protected holdout evaluation, forecast error "
            "analysis, model interpretation, packaging and "
            "reusable next-period forecasting."
        ),
        steps=steps,
        namespace={
            "np": np,
            "pd": pd,
            "json": json,
            "joblib": joblib,
            "Path": Path,
            "datetime": datetime,
            "timezone": timezone,
            "clone": clone,
            "time_series_data": (
                time_series_data
            ),
            "RANDOM_STATE": RANDOM_STATE,
            "N_SPLITS": N_SPLITS,
            "HOLDOUT_PERIODS": (
                HOLDOUT_PERIODS
            ),
            "SEASONAL_PERIOD": (
                SEASONAL_PERIOD
            ),
            "RandomForestRegressor": (
                RandomForestRegressor
            ),
            "TimeSeriesSplit": (
                TimeSeriesSplit
            ),
            "cross_val_score": (
                cross_val_score
            ),
            "GridSearchCV": (
                GridSearchCV
            ),
            "mean_absolute_error": (
                mean_absolute_error
            ),
            "mean_squared_error": (
                mean_squared_error
            ),
            "r2_score": r2_score,
        },
        preview=show_dataset,
    )