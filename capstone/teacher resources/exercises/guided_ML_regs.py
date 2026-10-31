import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from IPython.display import display

from sklearn.base import clone
from sklearn.datasets import load_diabetes
from sklearn.dummy import DummyRegressor
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.linear_model import Ridge
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import (
    GridSearchCV,
    KFold,
    cross_val_score,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from activity import GuidedActivity


# =========================================================
# CONFIGURATION
# =========================================================

RANDOM_STATE = 42
TEST_SIZE = 0.20
N_SPLITS = 5


# =========================================================
# DATASET
# =========================================================

diabetes = load_diabetes(as_frame=True)

regression_data = diabetes.frame.copy()


# =========================================================
# PREVIEW
# =========================================================

def show_dataset(namespace):
    data = namespace["regression_data"]

    print("Dataset available as: regression_data")
    print("Target column: target")
    print("Dataset shape:", data.shape)
    print("Random state:", namespace["RANDOM_STATE"])

    display(data.head())


# =========================================================
# VALIDATORS
# =========================================================

def validate_data_copy(namespace):
    if "data" not in namespace:
        return False

    result = namespace["data"]
    expected = namespace["regression_data"]

    return (
        isinstance(result, pd.DataFrame)
        and result.equals(expected)
        and result is not expected
    )


def validate_schema(namespace):
    if "schema_summary" not in namespace:
        return False

    result = namespace["schema_summary"]
    data = namespace["data"]

    if not isinstance(result, pd.DataFrame):
        return False

    required_columns = {
        "column",
        "dtype",
        "missing_values",
        "unique_values",
    }

    return (
        required_columns.issubset(result.columns)
        and len(result) == len(data.columns)
        and set(result["column"]) == set(data.columns)
    )


def validate_target(namespace):
    if "X" not in namespace or "y" not in namespace:
        return False

    X = namespace["X"]
    y = namespace["y"]
    data = namespace["data"]

    expected_X = data.drop(columns="target")
    expected_y = data["target"]

    return (
        isinstance(X, pd.DataFrame)
        and isinstance(y, pd.Series)
        and X.equals(expected_X)
        and y.equals(expected_y)
    )


def validate_duplicate_check(namespace):
    if "duplicate_count" not in namespace:
        return False

    expected = int(
        namespace["data"]
        .duplicated()
        .sum()
    )

    result = namespace["duplicate_count"]

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
            namespace["y"].mean(),
        )
        and np.isclose(
            result["median"],
            namespace["y"].median(),
        )
    )


def validate_split(namespace):
    required = {
        "X_train",
        "X_test",
        "y_train",
        "y_test",
    }

    if not required.issubset(namespace):
        return False

    X_train = namespace["X_train"]
    X_test = namespace["X_test"]
    y_train = namespace["y_train"]
    y_test = namespace["y_test"]

    return (
        isinstance(X_train, pd.DataFrame)
        and isinstance(X_test, pd.DataFrame)
        and isinstance(y_train, pd.Series)
        and isinstance(y_test, pd.Series)
        and len(X_train) + len(X_test)
        == len(namespace["X"])
        and len(X_train) == len(y_train)
        and len(X_test) == len(y_test)
        and set(X_train.index).isdisjoint(
            X_test.index
        )
    )


def validate_dummy_model(namespace):
    if "dummy_model" not in namespace:
        return False

    result = namespace["dummy_model"]

    return (
        isinstance(result, DummyRegressor)
        and result.strategy == "mean"
    )


def validate_cross_validator(namespace):
    if "cv" not in namespace:
        return False

    result = namespace["cv"]

    return (
        isinstance(result, KFold)
        and result.n_splits
        == namespace["N_SPLITS"]
        and result.shuffle is True
        and result.random_state
        == namespace["RANDOM_STATE"]
    )


def validate_dummy_scores(namespace):
    if "dummy_cv_scores" not in namespace:
        return False

    result = namespace["dummy_cv_scores"]

    return (
        isinstance(result, np.ndarray)
        and len(result)
        == namespace["N_SPLITS"]
        and np.all(result >= 0)
    )


def validate_pipeline(namespace):
    if "pipeline" not in namespace:
        return False

    result = namespace["pipeline"]

    if not isinstance(result, Pipeline):
        return False

    expected_steps = [
        "scaler",
        "selector",
        "regressor",
    ]

    return (
        list(result.named_steps.keys())
        == expected_steps
        and isinstance(
            result.named_steps["scaler"],
            StandardScaler,
        )
        and isinstance(
            result.named_steps["selector"],
            SelectKBest,
        )
        and isinstance(
            result.named_steps["regressor"],
            Ridge,
        )
    )


def validate_pipeline_scores(namespace):
    if "pipeline_cv_scores" not in namespace:
        return False

    result = namespace["pipeline_cv_scores"]

    return (
        isinstance(result, np.ndarray)
        and len(result)
        == namespace["N_SPLITS"]
        and np.all(result >= 0)
    )


def validate_baseline_comparison(namespace):
    if "baseline_comparison" not in namespace:
        return False

    result = namespace["baseline_comparison"]

    if not isinstance(result, pd.DataFrame):
        return False

    required_columns = {
        "model",
        "mean_rmse",
        "standard_deviation",
    }

    return (
        required_columns.issubset(
            result.columns
        )
        and len(result) == 2
        and set(result["model"])
        == {
            "Dummy baseline",
            "Ridge pipeline",
        }
    )


def validate_parameter_grid(namespace):
    if "parameter_grid" not in namespace:
        return False

    result = namespace["parameter_grid"]

    if not isinstance(result, dict):
        return False

    required = {
        "selector__k",
        "regressor__alpha",
    }

    return (
        required.issubset(result)
        and list(result["selector__k"])
        == [3, 5, 7, 10, "all"]
        and list(result["regressor__alpha"])
        == [
            0.01,
            0.1,
            1.0,
            10.0,
            100.0,
        ]
    )


def validate_grid_search(namespace):
    if "grid_search" not in namespace:
        return False

    result = namespace["grid_search"]

    return (
        isinstance(result, GridSearchCV)
        and result.scoring
        == "neg_root_mean_squared_error"
        and result.refit is True
        and result.cv is namespace["cv"]
    )


def validate_fitted_grid_search(namespace):
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

    result = namespace["tuning_results"]

    if not isinstance(result, pd.DataFrame):
        return False

    required = {
        "rank_test_score",
        "mean_test_score",
        "std_test_score",
        "params",
    }

    return (
        required.issubset(result.columns)
        and len(result) > 0
        and result.iloc[0][
            "rank_test_score"
        ] == 1
    )


def validate_best_pipeline(namespace):
    if "best_pipeline" not in namespace:
        return False

    result = namespace["best_pipeline"]

    return (
        isinstance(result, Pipeline)
        and result
        is namespace["grid_search"]
        .best_estimator_
    )


def validate_final_model(namespace):
    if "final_model" not in namespace:
        return False

    result = namespace["final_model"]

    if not isinstance(result, Pipeline):
        return False

    return (
        result is not namespace["best_pipeline"]
        and hasattr(
            result.named_steps["regressor"],
            "coef_",
        )
    )


def validate_test_predictions(namespace):
    if "test_predictions" not in namespace:
        return False

    result = namespace["test_predictions"]

    return (
        isinstance(result, np.ndarray)
        and result.ndim == 1
        and len(result)
        == len(namespace["y_test"])
    )


def validate_holdout_metrics(namespace):
    if "holdout_metrics" not in namespace:
        return False

    result = namespace["holdout_metrics"]

    if not isinstance(result, pd.Series):
        return False

    required = {
        "rmse",
        "mae",
        "r2",
    }

    if set(result.index) != required:
        return False

    return (
        result["rmse"] >= 0
        and result["mae"] >= 0
        and np.isfinite(
            result["r2"]
        )
    )


def validate_prediction_table(namespace):
    if "prediction_results" not in namespace:
        return False

    result = namespace["prediction_results"]

    if not isinstance(result, pd.DataFrame):
        return False

    required = {
        "actual",
        "predicted",
        "residual",
        "absolute_error",
    }

    return (
        required.issubset(
            result.columns
        )
        and len(result)
        == len(namespace["y_test"])
    )


def validate_residual_summary(namespace):
    if "residual_summary" not in namespace:
        return False

    result = namespace["residual_summary"]

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

    result = namespace["largest_errors"]

    return (
        isinstance(result, pd.DataFrame)
        and len(result) == 10
        and result[
            "absolute_error"
        ].is_monotonic_decreasing
    )


def validate_selected_features(namespace):
    if "selected_features" not in namespace:
        return False

    result = namespace["selected_features"]

    if not isinstance(result, list):
        return False

    selector = (
        namespace["final_model"]
        .named_steps["selector"]
    )

    expected = (
        namespace["X_train"]
        .columns[
            selector.get_support()
        ]
        .tolist()
    )

    return (
        result == expected
        and len(result) > 0
    )


def validate_feature_coefficients(namespace):
    if "feature_coefficients" not in namespace:
        return False

    result = namespace[
        "feature_coefficients"
    ]

    if not isinstance(result, pd.DataFrame):
        return False

    required = {
        "feature",
        "coefficient",
        "absolute_coefficient",
    }

    return (
        required.issubset(
            result.columns
        )
        and len(result)
        == len(
            namespace["selected_features"]
        )
        and result[
            "absolute_coefficient"
        ].is_monotonic_decreasing
    )


def validate_metadata(namespace):
    if "model_metadata" not in namespace:
        return False

    result = namespace["model_metadata"]

    if not isinstance(result, dict):
        return False

    required = {
        "model_type",
        "created_at_utc",
        "random_state",
        "test_size",
        "cross_validation_folds",
        "primary_metric",
        "best_parameters",
        "cross_validation_rmse",
        "holdout_metrics",
        "selected_features",
        "feature_columns",
    }

    return (
        required.issubset(result)
        and result["model_type"]
        == "Ridge regression pipeline"
        and result["primary_metric"]
        == "RMSE"
    )


def validate_model_bundle(namespace):
    required = {
        "model_bundle",
        "artifact_path",
    }

    if not required.issubset(namespace):
        return False

    bundle = namespace["model_bundle"]
    artifact_path = namespace[
        "artifact_path"
    ]

    if not isinstance(bundle, dict):
        return False

    if not {
        "model",
        "metadata",
    }.issubset(bundle):
        return False

    path = Path(artifact_path)

    if not path.exists():
        return False

    try:
        loaded = joblib.load(path)
    except Exception:
        return False

    return (
        isinstance(loaded, dict)
        and isinstance(
            loaded.get("model"),
            Pipeline,
        )
    )


def validate_prediction_function(namespace):
    if "predict_with_bundle" not in namespace:
        return False

    function = namespace[
        "predict_with_bundle"
    ]

    if not callable(function):
        return False

    sample = namespace[
        "X_test"
    ].iloc[[0]]

    try:
        result = function(
            namespace["model_bundle"],
            sample,
        )
    except Exception:
        return False

    return (
        isinstance(result, pd.DataFrame)
        and "prediction" in result.columns
        and len(result) == 1
    )


# =========================================================
# ACTIVITY STEPS
# =========================================================

steps = [

    # =====================================================
    # GROUP 1
    # =====================================================

    {
        "group": "1. Data understanding & quality",
        "title": "Step 1: Create a working copy",
        "instruction": (
            "Create an independent DataFrame called data by "
            "copying regression_data. Preserve the original "
            "source dataset."
        ),
        "placeholder": (
            "data = your_source_dataframe.copy()"
        ),
        "validator": validate_data_copy,
    },

    {
        "group": "1. Data understanding & quality",
        "title": "Step 2: Build a schema summary",
        "instruction": (
            "Create schema_summary with one row per column. "
            "Include column, dtype, missing_values and "
            "unique_values."
        ),
        "placeholder": (
            "schema_summary = pd.DataFrame(\n"
            "    {\n"
            '        "column": data.columns,\n'
            '        "dtype": data.dtypes.astype(str).values,\n'
            '        "missing_values": data.isna().sum().values,\n'
            '        "unique_values": data.nunique().values,\n'
            "    }\n"
            ")\n\n"
            "display(schema_summary)"
        ),
        "validator": validate_schema,
    },

    {
        "group": "1. Data understanding & quality",
        "title": "Step 3: Separate features and target",
        "instruction": (
            "Create X containing all predictor columns and y "
            "containing the target column. The target must not "
            "appear inside X."
        ),
        "placeholder": (
            "X = data.drop(columns=your_target_column)\n"
            "y = data[your_target_column]"
        ),
        "validator": validate_target,
    },

    {
        "group": "1. Data understanding & quality",
        "title": "Step 4: Check for duplicate records",
        "instruction": (
            "Create duplicate_count containing the number of "
            "duplicated rows in data."
        ),
        "placeholder": (
            "duplicate_count = int(\n"
            "    data.duplicated().sum()\n"
            ")\n\n"
            "print(duplicate_count)"
        ),
        "validator": validate_duplicate_check,
    },

    {
        "group": "1. Data understanding & quality",
        "title": "Step 5: Audit missing values",
        "instruction": (
            "Create missing_summary containing the number of "
            "missing values in each column, sorted from highest "
            "to lowest."
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
        "group": "1. Data understanding & quality",
        "title": "Step 6: Summarise the regression target",
        "instruction": (
            "Create target_summary containing the target mean, "
            "median, standard deviation, minimum and maximum."
        ),
        "placeholder": (
            "target_summary = pd.Series(\n"
            "    {\n"
            '        "mean": y.mean(),\n'
            '        "median": y.median(),\n'
            '        "standard_deviation": y.std(),\n'
            '        "minimum": y.min(),\n'
            '        "maximum": y.max(),\n'
            "    }\n"
            ")\n\n"
            "print(target_summary)"
        ),
        "validator": validate_target_summary,
    },

    # =====================================================
    # GROUP 2
    # =====================================================

    {
        "group": "2. Train-test strategy",
        "title": "Step 7: Create a protected holdout set",
        "instruction": (
            "Split X and y into training and test sets using "
            "TEST_SIZE and RANDOM_STATE. Keep the test data "
            "protected until final evaluation."
        ),
        "placeholder": (
            "X_train, X_test, y_train, y_test = (\n"
            "    train_test_split(\n"
            "        X,\n"
            "        y,\n"
            "        test_size=TEST_SIZE,\n"
            "        random_state=RANDOM_STATE,\n"
            "    )\n"
            ")"
        ),
        "validator": validate_split,
    },

    # =====================================================
    # GROUP 3
    # =====================================================

    {
        "group": "3. Baseline & validation",
        "title": "Step 8: Create a naive regression baseline",
        "instruction": (
            "Create dummy_model using DummyRegressor with the "
            "mean strategy. A trained model should outperform "
            "this simple reference point."
        ),
        "placeholder": (
            "dummy_model = DummyRegressor(\n"
            '    strategy="mean"\n'
            ")"
        ),
        "validator": validate_dummy_model,
    },

    {
        "group": "3. Baseline & validation",
        "title": "Step 9: Configure cross-validation",
        "instruction": (
            "Create a KFold object called cv using N_SPLITS, "
            "shuffle=True and random_state=RANDOM_STATE."
        ),
        "placeholder": (
            "cv = KFold(\n"
            "    n_splits=N_SPLITS,\n"
            "    shuffle=True,\n"
            "    random_state=RANDOM_STATE,\n"
            ")"
        ),
        "validator": validate_cross_validator,
    },

    {
        "group": "3. Baseline & validation",
        "title": "Step 10: Cross-validate the baseline",
        "instruction": (
            "Evaluate dummy_model on the training data using "
            "negative root mean squared error. Convert the "
            "returned scores to positive RMSE values and store "
            "them in dummy_cv_scores."
        ),
        "placeholder": (
            "dummy_cv_scores = -cross_val_score(\n"
            "    dummy_model,\n"
            "    X_train,\n"
            "    y_train,\n"
            "    cv=cv,\n"
            '    scoring="neg_root_mean_squared_error",\n'
            ")\n\n"
            "print(dummy_cv_scores)"
        ),
        "validator": validate_dummy_scores,
    },

    # =====================================================
    # GROUP 4
    # =====================================================

    {
        "group": "4. Regression pipeline",
        "title": "Step 11: Build a leakage-safe regression pipeline",
        "instruction": (
            "Create pipeline containing StandardScaler, "
            "SelectKBest using f_regression and Ridge regression. "
            "Name the steps scaler, selector and regressor."
        ),
        "placeholder": (
            "pipeline = Pipeline(\n"
            "    steps=[\n"
            "        (\n"
            '            "scaler",\n'
            "            StandardScaler(),\n"
            "        ),\n"
            "        (\n"
            '            "selector",\n'
            "            SelectKBest(\n"
            "                score_func=f_regression,\n"
            "                k=5,\n"
            "            ),\n"
            "        ),\n"
            "        (\n"
            '            "regressor",\n'
            "            Ridge(\n"
            "                alpha=1.0,\n"
            "            ),\n"
            "        ),\n"
            "    ]\n"
            ")"
        ),
        "validator": validate_pipeline,
        "input_height": "340px",
    },

    {
        "group": "4. Regression pipeline",
        "title": "Step 12: Cross-validate the regression pipeline",
        "instruction": (
            "Evaluate pipeline on X_train and y_train using "
            "RMSE. Store positive fold errors in "
            "pipeline_cv_scores."
        ),
        "placeholder": (
            "pipeline_cv_scores = -cross_val_score(\n"
            "    pipeline,\n"
            "    X_train,\n"
            "    y_train,\n"
            "    cv=cv,\n"
            '    scoring="neg_root_mean_squared_error",\n'
            ")\n\n"
            "print(pipeline_cv_scores)"
        ),
        "validator": validate_pipeline_scores,
    },

    {
        "group": "4. Regression pipeline",
        "title": "Step 13: Compare against the baseline",
        "instruction": (
            "Create baseline_comparison with one row for the "
            "dummy baseline and one for the Ridge pipeline. "
            "Include model, mean_rmse and standard_deviation."
        ),
        "placeholder": (
            "baseline_comparison = pd.DataFrame(\n"
            "    {\n"
            '        "model": [\n'
            '            "Dummy baseline",\n'
            '            "Ridge pipeline",\n'
            "        ],\n"
            '        "mean_rmse": [\n'
            "            dummy_cv_scores.mean(),\n"
            "            pipeline_cv_scores.mean(),\n"
            "        ],\n"
            '        "standard_deviation": [\n'
            "            dummy_cv_scores.std(),\n"
            "            pipeline_cv_scores.std(),\n"
            "        ],\n"
            "    }\n"
            ")\n\n"
            "display(baseline_comparison)"
        ),
        "validator": validate_baseline_comparison,
    },

    # =====================================================
    # GROUP 5
    # =====================================================

    {
        "group": "5. Hyperparameter tuning",
        "title": "Step 14: Define a hyperparameter search space",
        "instruction": (
            "Create parameter_grid containing candidate values "
            "for the number of selected features and Ridge alpha."
        ),
        "placeholder": (
            "parameter_grid = {\n"
            '    "selector__k": [\n'
            '        3, 5, 7, 10, "all"\n'
            "    ],\n"
            '    "regressor__alpha": [\n'
            "        0.01,\n"
            "        0.1,\n"
            "        1.0,\n"
            "        10.0,\n"
            "        100.0,\n"
            "    ],\n"
            "}"
        ),
        "validator": validate_parameter_grid,
    },

    {
        "group": "5. Hyperparameter tuning",
        "title": "Step 15: Configure grid search",
        "instruction": (
            "Create grid_search using GridSearchCV. Search the "
            "pipeline using parameter_grid and cv. Optimise for "
            "negative root mean squared error."
        ),
        "placeholder": (
            "grid_search = GridSearchCV(\n"
            "    estimator=pipeline,\n"
            "    param_grid=parameter_grid,\n"
            "    cv=cv,\n"
            '    scoring="neg_root_mean_squared_error",\n'
            "    refit=True,\n"
            "    return_train_score=True,\n"
            "    n_jobs=-1,\n"
            ")"
        ),
        "validator": validate_grid_search,
    },

    {
        "group": "5. Hyperparameter tuning",
        "title": "Step 16: Fit the search using training data only",
        "instruction": (
            "Fit grid_search using X_train and y_train. The "
            "protected test set must not participate in model "
            "selection."
        ),
        "placeholder": (
            "grid_search.fit(\n"
            "    X_train,\n"
            "    y_train,\n"
            ")\n\n"
            "print(grid_search.best_params_)\n"
            "print(grid_search.best_score_)"
        ),
        "validator": validate_fitted_grid_search,
    },

    {
        "group": "5. Hyperparameter tuning",
        "title": "Step 17: Review model-selection results",
        "instruction": (
            "Create tuning_results from grid_search.cv_results_. "
            "Keep rank_test_score, mean_test_score, "
            "std_test_score and params. Sort the best model first."
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
            ")\n\n"
            "display(tuning_results.head())"
        ),
        "validator": validate_tuning_results,
    },

    {
        "group": "5. Hyperparameter tuning",
        "title": "Step 18: Retrieve the selected pipeline",
        "instruction": (
            "Store grid_search.best_estimator_ in a variable "
            "called best_pipeline."
        ),
        "placeholder": (
            "best_pipeline = (\n"
            "    grid_search.best_estimator_\n"
            ")\n\n"
            "print(best_pipeline)"
        ),
        "validator": validate_best_pipeline,
    },

    # =====================================================
    # GROUP 6
    # =====================================================

    {
        "group": "6. Final model evaluation",
        "title": "Step 19: Refit the final model",
        "instruction": (
            "Clone best_pipeline, store the clone as final_model "
            "and fit it using all available training data."
        ),
        "placeholder": (
            "final_model = clone(\n"
            "    best_pipeline\n"
            ")\n\n"
            "final_model.fit(\n"
            "    X_train,\n"
            "    y_train,\n"
            ")"
        ),
        "validator": validate_final_model,
    },

    {
        "group": "6. Final model evaluation",
        "title": "Step 20: Predict the protected holdout set",
        "instruction": (
            "Use final_model to predict target values for X_test. "
            "Store the predictions in test_predictions."
        ),
        "placeholder": (
            "test_predictions = (\n"
            "    final_model.predict(X_test)\n"
            ")"
        ),
        "validator": validate_test_predictions,
    },

    {
        "group": "6. Final model evaluation",
        "title": "Step 21: Calculate holdout metrics",
        "instruction": (
            "Create holdout_metrics containing RMSE, MAE and R² "
            "for the protected test set."
        ),
        "placeholder": (
            "holdout_metrics = pd.Series(\n"
            "    {\n"
            '        "rmse": (\n'
            "            mean_squared_error(\n"
            "                y_test,\n"
            "                test_predictions,\n"
            "            ) ** 0.5\n"
            "        ),\n"
            '        "mae": mean_absolute_error(\n'
            "            y_test,\n"
            "            test_predictions,\n"
            "        ),\n"
            '        "r2": r2_score(\n'
            "            y_test,\n"
            "            test_predictions,\n"
            "        ),\n"
            "    }\n"
            ")\n\n"
            "print(holdout_metrics)"
        ),
        "validator": validate_holdout_metrics,
        "input_height": "300px",
    },

    # =====================================================
    # GROUP 7
    # =====================================================

    {
        "group": "7. Residual & error analysis",
        "title": "Step 22: Create a prediction results table",
        "instruction": (
            "Create prediction_results indexed like X_test. "
            "Include actual, predicted, residual and "
            "absolute_error."
        ),
        "placeholder": (
            "prediction_results = pd.DataFrame(\n"
            "    {\n"
            '        "actual": y_test,\n'
            '        "predicted": test_predictions,\n'
            "    },\n"
            "    index=X_test.index,\n"
            ")\n\n"
            'prediction_results["residual"] = (\n'
            '    prediction_results["actual"]\n'
            '    - prediction_results["predicted"]\n'
            ")\n\n"
            'prediction_results["absolute_error"] = (\n'
            '    prediction_results["residual"].abs()\n'
            ")\n\n"
            "display(prediction_results.head())"
        ),
        "validator": validate_prediction_table,
        "input_height": "330px",
    },

    {
        "group": "7. Residual & error analysis",
        "title": "Step 23: Summarise residual behaviour",
        "instruction": (
            "Create residual_summary containing mean residual, "
            "median residual, residual standard deviation and "
            "largest absolute error."
        ),
        "placeholder": (
            "residual_summary = pd.Series(\n"
            "    {\n"
            '        "mean_residual": (\n'
            '            prediction_results["residual"].mean()\n'
            "        ),\n"
            '        "median_residual": (\n'
            '            prediction_results["residual"].median()\n'
            "        ),\n"
            '        "residual_std": (\n'
            '            prediction_results["residual"].std()\n'
            "        ),\n"
            '        "largest_absolute_error": (\n'
            '            prediction_results["absolute_error"].max()\n'
            "        ),\n"
            "    }\n"
            ")\n\n"
            "print(residual_summary)"
        ),
        "validator": validate_residual_summary,
        "input_height": "300px",
    },

    {
        "group": "7. Residual & error analysis",
        "title": "Step 24: Inspect the largest prediction errors",
        "instruction": (
            "Create largest_errors containing the 10 observations "
            "with the highest absolute prediction error."
        ),
        "placeholder": (
            "largest_errors = (\n"
            "    prediction_results\n"
            "    .sort_values(\n"
            '        "absolute_error",\n'
            "        ascending=False,\n"
            "    )\n"
            "    .head(10)\n"
            ")\n\n"
            "display(largest_errors)"
        ),
        "validator": validate_largest_errors,
    },

    # =====================================================
    # GROUP 8
    # =====================================================

    {
        "group": "8. Model interpretation",
        "title": "Step 25: Recover selected feature names",
        "instruction": (
            "Access the fitted feature selector inside "
            "final_model and create selected_features containing "
            "the selected input column names."
        ),
        "placeholder": (
            "feature_mask = (\n"
            "    final_model\n"
            '    .named_steps["selector"]\n'
            "    .get_support()\n"
            ")\n\n"
            "selected_features = (\n"
            "    X_train.columns[\n"
            "        feature_mask\n"
            "    ]\n"
            "    .tolist()\n"
            ")\n\n"
            "print(selected_features)"
        ),
        "validator": validate_selected_features,
    },

    {
        "group": "8. Model interpretation",
        "title": "Step 26: Inspect Ridge coefficients",
        "instruction": (
            "Create feature_coefficients containing each selected "
            "feature, its fitted Ridge coefficient and its "
            "absolute coefficient. Sort by absolute coefficient "
            "from highest to lowest."
        ),
        "placeholder": (
            "coefficients = (\n"
            "    final_model\n"
            '    .named_steps["regressor"]\n'
            "    .coef_\n"
            ")\n\n"
            "feature_coefficients = (\n"
            "    pd.DataFrame(\n"
            "        {\n"
            '            "feature": selected_features,\n'
            '            "coefficient": coefficients,\n'
            "        }\n"
            "    )\n"
            "    .assign(\n"
            "        absolute_coefficient=lambda frame: (\n"
            '            frame["coefficient"].abs()\n'
            "        )\n"
            "    )\n"
            "    .sort_values(\n"
            '        "absolute_coefficient",\n'
            "        ascending=False,\n"
            "    )\n"
            "    .reset_index(drop=True)\n"
            ")\n\n"
            "display(feature_coefficients)"
        ),
        "validator": validate_feature_coefficients,
        "input_height": "400px",
    },

    # =====================================================
    # GROUP 9
    # =====================================================

    {
        "group": "9. Packaging & inference",
        "title": "Step 27: Create model metadata",
        "instruction": (
            "Create model_metadata documenting the model type, "
            "creation time, reproducibility settings, primary "
            "metric, selected parameters, cross-validation RMSE, "
            "holdout metrics, selected features and input columns."
        ),
        "placeholder": (
            "model_metadata = {\n"
            '    "model_type": (\n'
            '        "Ridge regression pipeline"\n'
            "    ),\n"
            '    "created_at_utc": (\n'
            "        datetime.now(timezone.utc)\n"
            "        .isoformat()\n"
            "    ),\n"
            '    "random_state": RANDOM_STATE,\n'
            '    "test_size": TEST_SIZE,\n'
            '    "cross_validation_folds": N_SPLITS,\n'
            '    "primary_metric": "RMSE",\n'
            '    "best_parameters": (\n'
            "        grid_search.best_params_\n"
            "    ),\n"
            '    "cross_validation_rmse": float(\n'
            "        -grid_search.best_score_\n"
            "    ),\n"
            '    "holdout_metrics": (\n'
            "        holdout_metrics.to_dict()\n"
            "    ),\n"
            '    "selected_features": (\n'
            "        selected_features\n"
            "    ),\n"
            '    "feature_columns": (\n'
            "        X.columns.tolist()\n"
            "    ),\n"
            "}\n\n"
            "print(\n"
            "    json.dumps(\n"
            "        model_metadata,\n"
            "        indent=2,\n"
            "    )\n"
            ")"
        ),
        "validator": validate_metadata,
        "input_height": "460px",
    },

    {
        "group": "9. Packaging & inference",
        "title": "Step 28: Package and save the model",
        "instruction": (
            "Create model_bundle containing final_model and "
            "model_metadata. Save it using joblib to "
            "diabetes_regression_pipeline.joblib and store the "
            "path in artifact_path."
        ),
        "placeholder": (
            "model_bundle = {\n"
            '    "model": final_model,\n'
            '    "metadata": model_metadata,\n'
            "}\n\n"
            "artifact_path = Path(\n"
            '    "diabetes_regression_pipeline.joblib"\n'
            ")\n\n"
            "joblib.dump(\n"
            "    model_bundle,\n"
            "    artifact_path,\n"
            ")\n\n"
            "print(artifact_path.resolve())"
        ),
        "validator": validate_model_bundle,
        "input_height": "280px",
    },

    {
        "group": "9. Packaging & inference",
        "title": "Step 29: Create a reusable prediction function",
        "instruction": (
            "Define predict_with_bundle(bundle, new_data). "
            "Validate that all required input columns are present, "
            "apply the saved pipeline and return a DataFrame "
            "containing the regression prediction."
        ),
        "placeholder": (
            "def predict_with_bundle(bundle, new_data):\n"
            "    model = bundle[\"model\"]\n"
            "    metadata = bundle[\"metadata\"]\n\n"
            "    expected_columns = metadata[\n"
            '        "feature_columns"\n'
            "    ]\n\n"
            "    missing_columns = set(\n"
            "        expected_columns\n"
            "    ) - set(new_data.columns)\n\n"
            "    if missing_columns:\n"
            "        raise ValueError(\n"
            '            "Missing required columns: "\n'
            "            f\"{sorted(missing_columns)}\"\n"
            "        )\n\n"
            "    ordered_data = new_data[\n"
            "        expected_columns\n"
            "    ]\n\n"
            "    predictions = model.predict(\n"
            "        ordered_data\n"
            "    )\n\n"
            "    return pd.DataFrame(\n"
            "        {\n"
            '            "prediction": predictions,\n'
            "        },\n"
            "        index=new_data.index,\n"
            "    )\n\n"
            "sample_prediction = predict_with_bundle(\n"
            "    model_bundle,\n"
            "    X_test.iloc[[0]],\n"
            ")\n\n"
            "display(sample_prediction)"
        ),
        "validator": validate_prediction_function,
        "input_height": "560px",
    },
]


# =========================================================
# CREATE ACTIVITY
# =========================================================

def create_activity():
    return GuidedActivity(
        title="Professional regression workflow",
        introduction=(
            "Develop a reproducible end-to-end machine-learning "
            "regression workflow using the diabetes dataset. "
            "The activity covers data-quality checks, a protected "
            "holdout set, baseline comparison, leakage-safe "
            "preprocessing, feature selection, cross-validation, "
            "hyperparameter tuning, final evaluation using RMSE, "
            "MAE and R², residual analysis, model interpretation, "
            "metadata capture, model packaging and reusable "
            "inference."
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
            "diabetes": diabetes,
            "regression_data": regression_data,
            "RANDOM_STATE": RANDOM_STATE,
            "TEST_SIZE": TEST_SIZE,
            "N_SPLITS": N_SPLITS,
            "DummyRegressor": DummyRegressor,
            "StandardScaler": StandardScaler,
            "SelectKBest": SelectKBest,
            "f_regression": f_regression,
            "Ridge": Ridge,
            "Pipeline": Pipeline,
            "train_test_split": train_test_split,
            "KFold": KFold,
            "cross_val_score": cross_val_score,
            "GridSearchCV": GridSearchCV,
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