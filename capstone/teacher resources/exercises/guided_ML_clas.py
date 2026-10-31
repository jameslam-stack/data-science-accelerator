import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from IPython.display import display
from sklearn.base import clone
from sklearn.datasets import load_breast_cancer
from sklearn.dummy import DummyClassifier
from sklearn.inspection import permutation_importance
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    cross_val_predict,
    cross_val_score,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from activity import GuidedActivity


RANDOM_STATE = 42
TEST_SIZE = 0.20
N_SPLITS = 5

cancer = load_breast_cancer(as_frame=True)

classification_data = cancer.frame.copy()


def show_dataset(namespace):
    data = namespace["classification_data"]

    print("Dataset available as: classification_data")
    print("Target column: target")
    print("Target names:", list(namespace["cancer"].target_names))
    print("Dataset shape:", data.shape)
    print("Random state:", namespace["RANDOM_STATE"])

    display(data.head())


def validate_data_copy(namespace):
    if "data" not in namespace:
        return False

    result = namespace["data"]
    expected = namespace["classification_data"]

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

    if not required_columns.issubset(result.columns):
        return False

    return (
        len(result) == len(data.columns)
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

    result = namespace["duplicate_count"]
    expected = int(namespace["data"].duplicated().sum())

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


def validate_class_distribution(namespace):
    required_variables = {
        "class_counts",
        "class_proportions",
    }

    if not required_variables.issubset(namespace):
        return False

    class_counts = namespace["class_counts"]
    class_proportions = namespace["class_proportions"]
    y = namespace["y"]

    expected_counts = y.value_counts().sort_index()

    expected_proportions = (
        y.value_counts(normalize=True)
        .sort_index()
    )

    return (
        isinstance(class_counts, pd.Series)
        and isinstance(class_proportions, pd.Series)
        and class_counts.equals(expected_counts)
        and np.allclose(
            class_proportions.values,
            expected_proportions.values,
        )
    )


def validate_split(namespace):
    required_variables = {
        "X_train",
        "X_test",
        "y_train",
        "y_test",
    }

    if not required_variables.issubset(namespace):
        return False

    X_train = namespace["X_train"]
    X_test = namespace["X_test"]
    y_train = namespace["y_train"]
    y_test = namespace["y_test"]

    if not (
        isinstance(X_train, pd.DataFrame)
        and isinstance(X_test, pd.DataFrame)
        and isinstance(y_train, pd.Series)
        and isinstance(y_test, pd.Series)
    ):
        return False

    total_rows = len(namespace["X"])

    return (
        len(X_train) + len(X_test) == total_rows
        and len(X_train) == len(y_train)
        and len(X_test) == len(y_test)
        and len(X_test) == 114
        and set(X_train.index).isdisjoint(X_test.index)
    )


def validate_train_distribution(namespace):
    if "training_distribution" not in namespace:
        return False

    result = namespace["training_distribution"]

    expected = (
        namespace["y_train"]
        .value_counts(normalize=True)
        .sort_index()
    )

    return (
        isinstance(result, pd.Series)
        and np.allclose(
            result.sort_index().values,
            expected.values,
        )
    )


def validate_dummy_model(namespace):
    if "dummy_model" not in namespace:
        return False

    result = namespace["dummy_model"]

    return (
        isinstance(result, DummyClassifier)
        and result.strategy == "most_frequent"
    )


def validate_cross_validator(namespace):
    if "cv" not in namespace:
        return False

    result = namespace["cv"]

    return (
        isinstance(result, StratifiedKFold)
        and result.n_splits == namespace["N_SPLITS"]
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
        and len(result) == namespace["N_SPLITS"]
        and np.all(result >= 0)
        and np.all(result <= 1)
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
        "classifier",
    ]

    return (
        list(result.named_steps.keys()) == expected_steps
        and isinstance(
            result.named_steps["scaler"],
            StandardScaler,
        )
        and isinstance(
            result.named_steps["selector"],
            SelectKBest,
        )
        and isinstance(
            result.named_steps["classifier"],
            LogisticRegression,
        )
    )


def validate_pipeline_scores(namespace):
    if "pipeline_cv_scores" not in namespace:
        return False

    result = namespace["pipeline_cv_scores"]

    return (
        isinstance(result, np.ndarray)
        and len(result) == namespace["N_SPLITS"]
        and np.all(result >= 0)
        and np.all(result <= 1)
    )


def validate_baseline_comparison(namespace):
    if "baseline_comparison" not in namespace:
        return False

    result = namespace["baseline_comparison"]

    if not isinstance(result, pd.DataFrame):
        return False

    required_columns = {
        "model",
        "mean_roc_auc",
        "standard_deviation",
    }

    if not required_columns.issubset(result.columns):
        return False

    return (
        set(result["model"])
        == {"Dummy classifier", "Logistic pipeline"}
        and len(result) == 2
    )


def validate_parameter_grid(namespace):
    if "parameter_grid" not in namespace:
        return False

    result = namespace["parameter_grid"]

    if not isinstance(result, dict):
        return False

    required_parameters = {
        "selector__k",
        "classifier__C",
        "classifier__class_weight",
    }

    if not required_parameters.issubset(result):
        return False

    return (
        list(result["selector__k"])
        == [5, 10, 15, 20, "all"]
        and list(result["classifier__C"])
        == [0.01, 0.1, 1, 10, 100]
        and list(result["classifier__class_weight"])
        == [None, "balanced"]
    )


def validate_grid_search(namespace):
    if "grid_search" not in namespace:
        return False

    result = namespace["grid_search"]

    return (
        isinstance(result, GridSearchCV)
        and result.scoring == "roc_auc"
        and result.refit is True
        and result.cv is namespace["cv"]
        and result.return_train_score is True
    )


def validate_fitted_grid_search(namespace):
    if "grid_search" not in namespace:
        return False

    result = namespace["grid_search"]

    return (
        isinstance(result, GridSearchCV)
        and hasattr(result, "best_estimator_")
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

    required_columns = {
        "rank_test_score",
        "mean_test_score",
        "std_test_score",
        "mean_train_score",
        "params",
    }

    return (
        required_columns.issubset(result.columns)
        and len(result) > 0
        and result.iloc[0]["rank_test_score"] == 1
    )


def validate_best_pipeline(namespace):
    if "best_pipeline" not in namespace:
        return False

    result = namespace["best_pipeline"]

    return (
        isinstance(result, Pipeline)
        and result is namespace["grid_search"].best_estimator_
        and hasattr(
            result.named_steps["classifier"],
            "coef_",
        )
    )


def validate_oof_probabilities(namespace):
    if "oof_probabilities" not in namespace:
        return False

    result = namespace["oof_probabilities"]

    return (
        isinstance(result, np.ndarray)
        and result.ndim == 1
        and len(result) == len(namespace["y_train"])
        and np.all(result >= 0)
        and np.all(result <= 1)
    )


def validate_threshold_table(namespace):
    if "threshold_results" not in namespace:
        return False

    result = namespace["threshold_results"]

    if not isinstance(result, pd.DataFrame):
        return False

    required_columns = {
        "threshold",
        "precision",
        "recall",
        "f1",
    }

    if not required_columns.issubset(result.columns):
        return False

    return (
        len(result) > 0
        and result["threshold"].between(0, 1).all()
        and result["precision"].between(0, 1).all()
        and result["recall"].between(0, 1).all()
        and result["f1"].between(0, 1).all()
    )


def validate_threshold(namespace):
    if "decision_threshold" not in namespace:
        return False

    result = namespace["decision_threshold"]

    return (
        isinstance(result, (float, np.floating))
        and 0 <= float(result) <= 1
    )


def validate_refitted_model(namespace):
    if "final_model" not in namespace:
        return False

    result = namespace["final_model"]

    if not isinstance(result, Pipeline):
        return False

    classifier = result.named_steps["classifier"]

    return (
        result is not namespace["best_pipeline"]
        and hasattr(classifier, "coef_")
        and hasattr(classifier, "classes_")
    )


def validate_test_probabilities(namespace):
    if "test_probabilities" not in namespace:
        return False

    result = namespace["test_probabilities"]

    return (
        isinstance(result, np.ndarray)
        and result.ndim == 1
        and len(result) == len(namespace["X_test"])
        and np.all(result >= 0)
        and np.all(result <= 1)
    )


def validate_threshold_predictions(namespace):
    if "test_predictions" not in namespace:
        return False

    result = namespace["test_predictions"]

    expected = (
        namespace["test_probabilities"]
        >= namespace["decision_threshold"]
    ).astype(int)

    return (
        isinstance(result, np.ndarray)
        and np.array_equal(result, expected)
    )


def validate_holdout_metrics(namespace):
    if "holdout_metrics" not in namespace:
        return False

    result = namespace["holdout_metrics"]

    if not isinstance(result, pd.Series):
        return False

    expected_metrics = {
        "accuracy",
        "balanced_accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc",
    }

    if set(result.index) != expected_metrics:
        return False

    return result.between(0, 1).all()


def validate_confusion(namespace):
    if "holdout_confusion_matrix" not in namespace:
        return False

    result = namespace["holdout_confusion_matrix"]

    expected = confusion_matrix(
        namespace["y_test"],
        namespace["test_predictions"],
    )

    return (
        isinstance(result, np.ndarray)
        and result.shape == (2, 2)
        and np.array_equal(result, expected)
    )


def validate_report(namespace):
    if "holdout_report" not in namespace:
        return False

    result = namespace["holdout_report"]

    return (
        isinstance(result, str)
        and "precision" in result
        and "recall" in result
        and "f1-score" in result
    )


def validate_selected_features(namespace):
    if "selected_features" not in namespace:
        return False

    result = namespace["selected_features"]

    if not isinstance(result, list):
        return False

    selector = namespace["final_model"].named_steps[
        "selector"
    ]

    expected = (
        namespace["X_train"]
        .columns[selector.get_support()]
        .tolist()
    )

    return (
        result == expected
        and len(result) > 0
    )


def validate_feature_coefficients(namespace):
    if "feature_coefficients" not in namespace:
        return False

    result = namespace["feature_coefficients"]

    if not isinstance(result, pd.DataFrame):
        return False

    required_columns = {
        "feature",
        "coefficient",
        "absolute_coefficient",
    }

    return (
        required_columns.issubset(result.columns)
        and len(result)
        == len(namespace["selected_features"])
        and result["absolute_coefficient"]
        .is_monotonic_decreasing
    )


def validate_permutation_importance(namespace):
    if "permutation_results" not in namespace:
        return False

    result = namespace["permutation_results"]

    if not isinstance(result, pd.DataFrame):
        return False

    required_columns = {
        "feature",
        "importance_mean",
        "importance_std",
    }

    return (
        required_columns.issubset(result.columns)
        and len(result) == namespace["X_test"].shape[1]
        and result["importance_mean"]
        .is_monotonic_decreasing
    )


def validate_error_analysis(namespace):
    if "error_analysis" not in namespace:
        return False

    result = namespace["error_analysis"]

    if not isinstance(result, pd.DataFrame):
        return False

    required_columns = {
        "actual",
        "predicted",
        "probability",
        "correct",
    }

    if not required_columns.issubset(result.columns):
        return False

    return (
        len(result) == len(namespace["y_test"])
        and result["correct"].dtype == bool
    )


def validate_model_metadata(namespace):
    if "model_metadata" not in namespace:
        return False

    result = namespace["model_metadata"]

    if not isinstance(result, dict):
        return False

    required_keys = {
        "model_type",
        "created_at_utc",
        "random_state",
        "test_size",
        "cross_validation_folds",
        "best_parameters",
        "cross_validation_roc_auc",
        "holdout_metrics",
        "decision_threshold",
        "selected_features",
        "feature_columns",
        "target_names",
    }

    return (
        required_keys.issubset(result)
        and result["model_type"]
        == "LogisticRegression pipeline"
        and result["random_state"]
        == namespace["RANDOM_STATE"]
    )


def validate_model_bundle(namespace):
    required_variables = {
        "model_bundle",
        "artifact_path",
    }

    if not required_variables.issubset(namespace):
        return False

    model_bundle = namespace["model_bundle"]
    artifact_path = namespace["artifact_path"]

    if not isinstance(model_bundle, dict):
        return False

    required_keys = {
        "model",
        "threshold",
        "metadata",
    }

    if not required_keys.issubset(model_bundle):
        return False

    path = Path(artifact_path)

    if not path.exists():
        return False

    try:
        loaded_bundle = joblib.load(path)
    except Exception:
        return False

    return (
        isinstance(loaded_bundle, dict)
        and isinstance(
            loaded_bundle.get("model"),
            Pipeline,
        )
        and loaded_bundle.get("threshold")
        == namespace["decision_threshold"]
    )


def validate_prediction_function(namespace):
    if "predict_with_bundle" not in namespace:
        return False

    prediction_function = namespace[
        "predict_with_bundle"
    ]

    if not callable(prediction_function):
        return False

    sample = namespace["X_test"].iloc[[0]]

    try:
        result = prediction_function(
            namespace["model_bundle"],
            sample,
        )
    except Exception:
        return False

    if not isinstance(result, pd.DataFrame):
        return False

    required_columns = {
        "prediction",
        "prediction_name",
        "probability",
        "threshold",
    }

    return (
        required_columns.issubset(result.columns)
        and len(result) == 1
    )


steps = [
    {
        "group": "1. Data preparation & quality",
        "title": "Step 1: Create a working copy",
        "instruction": (
            "Create an independent DataFrame called data by "
            "copying classification_data. Professional workflows "
            "avoid modifying the original source data directly."
        ),
        "placeholder": (
            "data = your_source_dataframe.copy()"
        ),
        "validator": validate_data_copy,
    },
    {
        "group": "1. Data preparation & quality",
        "title": "Step 2: Build a schema summary",
        "instruction": (
            "Create a DataFrame called schema_summary with one row "
            "per column. Include column, dtype, missing_values and "
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
        "group": "1. Data preparation & quality",
        "title": "Step 3: Separate features and target",
        "instruction": (
            "Create X containing every predictor column and y "
            "containing the target column. Do not include target "
            "inside X."
        ),
        "placeholder": (
            "X = data.drop(columns=your_target_column)\n"
            "y = data[your_target_column]"
        ),
        "validator": validate_target,
    },
    {
        "group": "1. Data preparation & quality",
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
        "group": "1. Data preparation & quality",
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
        "group": "1. Data preparation & quality",
        "title": "Step 6: Inspect class balance",
        "instruction": (
            "Create class_counts containing target counts and "
            "class_proportions containing normalised target "
            "frequencies. Sort both by class index."
        ),
        "placeholder": (
            "class_counts = (\n"
            "    y.value_counts().sort_index()\n"
            ")\n\n"
            "class_proportions = (\n"
            "    y\n"
            "    .value_counts(normalize=True)\n"
            "    .sort_index()\n"
            ")\n\n"
            "print(class_counts)\n"
            "print(class_proportions)"
        ),
        "validator": validate_class_distribution,
    },
    {
        "group": "1. Data preparation & quality",
        "title": "Step 7: Create a protected holdout set",
        "instruction": (
            "Split X and y into training and test sets. Use "
            "test_size=TEST_SIZE, random_state=RANDOM_STATE and "
            "stratify=y. The test set must remain untouched until "
            "the final evaluation."
        ),
        "placeholder": (
            "X_train, X_test, y_train, y_test = (\n"
            "    train_test_split(\n"
            "        X,\n"
            "        y,\n"
            "        test_size=TEST_SIZE,\n"
            "        random_state=RANDOM_STATE,\n"
            "        stratify=y,\n"
            "    )\n"
            ")"
        ),
        "validator": validate_split,
    },
    {
        "group": "1. Data preparation & quality",
        "title": "Step 8: Verify training-set class balance",
        "instruction": (
            "Create training_distribution containing the "
            "normalised class frequencies in y_train, sorted by "
            "class index."
        ),
        "placeholder": (
            "training_distribution = (\n"
            "    y_train\n"
            "    .value_counts(normalize=True)\n"
            "    .sort_index()\n"
            ")\n\n"
            "print(training_distribution)"
        ),
        "validator": validate_train_distribution,
    },
    {
        "group": "2. Baseline & pipeline design",
        "title": "Step 9: Create a naive baseline",
        "instruction": (
            "Create a DummyClassifier called dummy_model using "
            "the most_frequent strategy. A production model should "
            "demonstrably outperform a simple baseline."
        ),
        "placeholder": (
            "dummy_model = DummyClassifier(\n"
            '    strategy="most_frequent"\n'
            ")"
        ),
        "validator": validate_dummy_model,
    },
    {
        "group": "2. Baseline & pipeline design",
        "title": "Step 10: Configure stratified cross-validation",
        "instruction": (
            "Create a StratifiedKFold object called cv using "
            "N_SPLITS folds, shuffle=True and "
            "random_state=RANDOM_STATE."
        ),
        "placeholder": (
            "cv = StratifiedKFold(\n"
            "    n_splits=N_SPLITS,\n"
            "    shuffle=True,\n"
            "    random_state=RANDOM_STATE,\n"
            ")"
        ),
        "validator": validate_cross_validator,
    },
    {
        "group": "2. Baseline & pipeline design",
        "title": "Step 11: Cross-validate the baseline",
        "instruction": (
            "Evaluate dummy_model on X_train and y_train using "
            "cross_val_score, cv=cv and scoring='roc_auc'. Store "
            "the results in dummy_cv_scores."
        ),
        "placeholder": (
            "dummy_cv_scores = cross_val_score(\n"
            "    dummy_model,\n"
            "    X_train,\n"
            "    y_train,\n"
            "    cv=cv,\n"
            '    scoring="roc_auc",\n'
            ")\n\n"
            "print(dummy_cv_scores)"
        ),
        "validator": validate_dummy_scores,
    },
    {
        "group": "2. Baseline & pipeline design",
        "title": "Step 12: Build a leakage-safe pipeline",
        "instruction": (
            "Create pipeline with StandardScaler, SelectKBest "
            "using f_classif and LogisticRegression. Name the "
            "steps scaler, selector and classifier. Set the "
            "selector to k=10 and the classifier to max_iter=5000 "
            "and random_state=RANDOM_STATE."
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
            "                score_func=f_classif,\n"
            "                k=10,\n"
            "            ),\n"
            "        ),\n"
            "        (\n"
            '            "classifier",\n'
            "            LogisticRegression(\n"
            "                max_iter=5000,\n"
            "                random_state=RANDOM_STATE,\n"
            "            ),\n"
            "        ),\n"
            "    ]\n"
            ")"
        ),
        "validator": validate_pipeline,
        "input_height": "340px",
    },
    {
        "group": "2. Baseline & pipeline design",
        "title": "Step 13: Cross-validate the pipeline",
        "instruction": (
            "Evaluate pipeline on the training data using ROC AUC. "
            "Store the fold scores in pipeline_cv_scores. Because "
            "preprocessing is inside the pipeline, each fold fits "
            "its own scaler and feature selector."
        ),
        "placeholder": (
            "pipeline_cv_scores = cross_val_score(\n"
            "    pipeline,\n"
            "    X_train,\n"
            "    y_train,\n"
            "    cv=cv,\n"
            '    scoring="roc_auc",\n'
            ")\n\n"
            "print(pipeline_cv_scores)"
        ),
        "validator": validate_pipeline_scores,
    },
    {
        "group": "2. Baseline & pipeline design",
        "title": "Step 14: Compare against the baseline",
        "instruction": (
            "Create baseline_comparison with one row for the dummy "
            "classifier and one for the logistic pipeline. Include "
            "model, mean_roc_auc and standard_deviation."
        ),
        "placeholder": (
            "baseline_comparison = pd.DataFrame(\n"
            "    {\n"
            '        "model": [\n'
            '            "Dummy classifier",\n'
            '            "Logistic pipeline",\n'
            "        ],\n"
            '        "mean_roc_auc": [\n'
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
    {
        "group": "3. Hyperparameter tuning",
        "title": "Step 15: Define a hyperparameter search space",
        "instruction": (
            "Create parameter_grid containing candidate values "
            "for selector__k, classifier__C and "
            "classifier__class_weight."
        ),
        "placeholder": (
            "parameter_grid = {\n"
            '    "selector__k": [\n'
            '        5, 10, 15, 20, "all"\n'
            "    ],\n"
            '    "classifier__C": [\n'
            "        0.01, 0.1, 1, 10, 100\n"
            "    ],\n"
            '    "classifier__class_weight": [\n'
            '        None, "balanced"\n'
            "    ],\n"
            "}"
        ),
        "validator": validate_parameter_grid,
    },
    {
        "group": "3. Hyperparameter tuning",
        "title": "Step 16: Configure grid search",
        "instruction": (
            "Create grid_search using GridSearchCV. Search the "
            "pipeline with parameter_grid, cv=cv, "
            "scoring='roc_auc', refit=True, "
            "return_train_score=True and n_jobs=-1."
        ),
        "placeholder": (
            "grid_search = GridSearchCV(\n"
            "    estimator=pipeline,\n"
            "    param_grid=parameter_grid,\n"
            "    cv=cv,\n"
            '    scoring="roc_auc",\n'
            "    refit=True,\n"
            "    return_train_score=True,\n"
            "    n_jobs=-1,\n"
            ")"
        ),
        "validator": validate_grid_search,
    },
    {
        "group": "3. Hyperparameter tuning",
        "title": "Step 17: Fit the search on training data only",
        "instruction": (
            "Fit grid_search using X_train and y_train. Do not use "
            "the protected test data during model selection."
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
        "group": "3. Hyperparameter tuning",
        "title": "Step 18: Review model-selection results",
        "instruction": (
            "Create tuning_results from grid_search.cv_results_. "
            "Keep rank_test_score, mean_test_score, std_test_score, "
            "mean_train_score and params. Sort by rank_test_score."
        ),
        "placeholder": (
            "tuning_results = (\n"
            "    pd.DataFrame(grid_search.cv_results_)\n"
            "    [[\n"
            '        "rank_test_score",\n'
            '        "mean_test_score",\n'
            '        "std_test_score",\n'
            '        "mean_train_score",\n'
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
        "group": "3. Hyperparameter tuning",
        "title": "Step 19: Retrieve the selected pipeline",
        "instruction": (
            "Store grid_search.best_estimator_ in a variable "
            "called best_pipeline."
        ),
        "placeholder": (
            "best_pipeline = grid_search.best_estimator_\n\n"
            "print(best_pipeline)"
        ),
        "validator": validate_best_pipeline,
    },
    {
        "group": "4. Threshold optimisation",
        "title": "Step 20: Generate out-of-fold probabilities",
        "instruction": (
            "Use cross_val_predict with a clone of best_pipeline "
            "to generate training-set probabilities without "
            "predicting rows used to fit the same fold. Use "
            "method='predict_proba' and keep the positive-class "
            "probabilities in oof_probabilities."
        ),
        "placeholder": (
            "oof_probability_matrix = cross_val_predict(\n"
            "    clone(best_pipeline),\n"
            "    X_train,\n"
            "    y_train,\n"
            "    cv=cv,\n"
            '    method="predict_proba",\n'
            "    n_jobs=-1,\n"
            ")\n\n"
            "oof_probabilities = (\n"
            "    oof_probability_matrix[:, 1]\n"
            ")"
        ),
        "validator": validate_oof_probabilities,
        "input_height": "240px",
    },
    {
        "group": "4. Threshold optimisation",
        "title": "Step 21: Evaluate candidate decision thresholds",
        "instruction": (
            "Use precision_recall_curve with y_train and "
            "oof_probabilities. Create threshold_results containing "
            "threshold, precision, recall and f1 for each available "
            "threshold."
        ),
        "placeholder": (
            "precision_values, recall_values, thresholds = (\n"
            "    precision_recall_curve(\n"
            "        y_train,\n"
            "        oof_probabilities,\n"
            "    )\n"
            ")\n\n"
            "precision_values = precision_values[:-1]\n"
            "recall_values = recall_values[:-1]\n\n"
            "f1_values = (\n"
            "    2\n"
            "    * precision_values\n"
            "    * recall_values\n"
            "    / (\n"
            "        precision_values\n"
            "        + recall_values\n"
            "        + 1e-12\n"
            "    )\n"
            ")\n\n"
            "threshold_results = pd.DataFrame(\n"
            "    {\n"
            '        "threshold": thresholds,\n'
            '        "precision": precision_values,\n'
            '        "recall": recall_values,\n'
            '        "f1": f1_values,\n'
            "    }\n"
            ")\n\n"
            "display(threshold_results.head())"
        ),
        "validator": validate_threshold_table,
        "input_height": "380px",
    },
    {
        "group": "4. Threshold optimisation",
        "title": "Step 22: Select a decision threshold",
        "instruction": (
            "Select the threshold with the highest out-of-fold F1 "
            "score and store it as a float called "
            "decision_threshold."
        ),
        "placeholder": (
            "best_threshold_index = (\n"
            '    threshold_results["f1"].idxmax()\n'
            ")\n\n"
            "decision_threshold = float(\n"
            "    threshold_results.loc[\n"
            "        best_threshold_index,\n"
            '        "threshold",\n'
            "    ]\n"
            ")\n\n"
            "print(decision_threshold)"
        ),
        "validator": validate_threshold,
    },
    {
        "group": "5. Final model evaluation",
        "title": "Step 23: Refit the final model",
        "instruction": (
            "Clone best_pipeline, store the clone as final_model "
            "and fit it using all available training data."
        ),
        "placeholder": (
            "final_model = clone(best_pipeline)\n\n"
            "final_model.fit(\n"
            "    X_train,\n"
            "    y_train,\n"
            ")"
        ),
        "validator": validate_refitted_model,
    },
    {
        "group": "5. Final model evaluation",
        "title": "Step 24: Predict holdout probabilities",
        "instruction": (
            "Use final_model to calculate positive-class "
            "probabilities for X_test. Store the one-dimensional "
            "array in test_probabilities."
        ),
        "placeholder": (
            "test_probabilities = (\n"
            "    final_model\n"
            "    .predict_proba(X_test)\n"
            "    [:, 1]\n"
            ")"
        ),
        "validator": validate_test_probabilities,
    },
    {
        "group": "5. Final model evaluation",
        "title": "Step 25: Apply the selected threshold",
        "instruction": (
            "Create test_predictions by comparing "
            "test_probabilities with decision_threshold and "
            "converting the Boolean result to integers."
        ),
        "placeholder": (
            "test_predictions = (\n"
            "    test_probabilities\n"
            "    >= decision_threshold\n"
            ").astype(int)"
        ),
        "validator": validate_threshold_predictions,
    },
    {
        "group": "5. Final model evaluation",
        "title": "Step 26: Calculate holdout metrics",
        "instruction": (
            "Create a Series called holdout_metrics containing "
            "accuracy, balanced_accuracy, precision, recall, F1 "
            "and ROC AUC."
        ),
        "placeholder": (
            "holdout_metrics = pd.Series(\n"
            "    {\n"
            '        "accuracy": accuracy_score(\n'
            "            y_test,\n"
            "            test_predictions,\n"
            "        ),\n"
            '        "balanced_accuracy": (\n'
            "            balanced_accuracy_score(\n"
            "                y_test,\n"
            "                test_predictions,\n"
            "            )\n"
            "        ),\n"
            '        "precision": precision_score(\n'
            "            y_test,\n"
            "            test_predictions,\n"
            "        ),\n"
            '        "recall": recall_score(\n'
            "            y_test,\n"
            "            test_predictions,\n"
            "        ),\n"
            '        "f1": f1_score(\n'
            "            y_test,\n"
            "            test_predictions,\n"
            "        ),\n"
            '        "roc_auc": roc_auc_score(\n'
            "            y_test,\n"
            "            test_probabilities,\n"
            "        ),\n"
            "    }\n"
            ")\n\n"
            "print(holdout_metrics)"
        ),
        "validator": validate_holdout_metrics,
        "input_height": "420px",
    },
    {
        "group": "5. Final model evaluation",
        "title": "Step 27: Create a confusion matrix",
        "instruction": (
            "Create holdout_confusion_matrix using y_test and "
            "test_predictions."
        ),
        "placeholder": (
            "holdout_confusion_matrix = confusion_matrix(\n"
            "    y_test,\n"
            "    test_predictions,\n"
            ")\n\n"
            "print(holdout_confusion_matrix)"
        ),
        "validator": validate_confusion,
    },
    {
        "group": "5. Final model evaluation",
        "title": "Step 28: Create a classification report",
        "instruction": (
            "Create a string called holdout_report using "
            "classification_report. Pass the dataset target names "
            "and set zero_division=0."
        ),
        "placeholder": (
            "holdout_report = classification_report(\n"
            "    y_test,\n"
            "    test_predictions,\n"
            "    target_names=cancer.target_names,\n"
            "    zero_division=0,\n"
            ")\n\n"
            "print(holdout_report)"
        ),
        "validator": validate_report,
    },
    {
        "group": "6. Interpretability & error analysis",
        "title": "Step 29: Recover selected feature names",
        "instruction": (
            "Access the fitted selector inside final_model and "
            "create selected_features containing the selected "
            "column names."
        ),
        "placeholder": (
            "feature_mask = (\n"
            "    final_model\n"
            '    .named_steps["selector"]\n'
            "    .get_support()\n"
            ")\n\n"
            "selected_features = (\n"
            "    X_train.columns[feature_mask]\n"
            "    .tolist()\n"
            ")\n\n"
            "print(selected_features)"
        ),
        "validator": validate_selected_features,
    },
    {
        "group": "6. Interpretability & error analysis",
        "title": "Step 30: Inspect logistic coefficients",
        "instruction": (
            "Create feature_coefficients containing each selected "
            "feature, its fitted logistic-regression coefficient "
            "and its absolute coefficient. Sort from highest to "
            "lowest absolute coefficient."
        ),
        "placeholder": (
            "coefficients = (\n"
            "    final_model\n"
            '    .named_steps["classifier"]\n'
            "    .coef_[0]\n"
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
    {
        "group": "6. Interpretability & error analysis",
        "title": "Step 31: Calculate permutation importance",
        "instruction": (
            "Calculate permutation importance for final_model on "
            "the protected test set using scoring='roc_auc', "
            "n_repeats=20 and random_state=RANDOM_STATE. Create "
            "permutation_results with feature, importance_mean and "
            "importance_std, sorted by mean importance."
        ),
        "placeholder": (
            "importance = permutation_importance(\n"
            "    final_model,\n"
            "    X_test,\n"
            "    y_test,\n"
            '    scoring="roc_auc",\n'
            "    n_repeats=20,\n"
            "    random_state=RANDOM_STATE,\n"
            "    n_jobs=-1,\n"
            ")\n\n"
            "permutation_results = (\n"
            "    pd.DataFrame(\n"
            "        {\n"
            '            "feature": X_test.columns,\n'
            '            "importance_mean": (\n'
            "                importance.importances_mean\n"
            "            ),\n"
            '            "importance_std": (\n'
            "                importance.importances_std\n"
            "            ),\n"
            "        }\n"
            "    )\n"
            "    .sort_values(\n"
            '        "importance_mean",\n'
            "        ascending=False,\n"
            "    )\n"
            "    .reset_index(drop=True)\n"
            ")\n\n"
            "display(permutation_results.head(10))"
        ),
        "validator": validate_permutation_importance,
        "input_height": "430px",
    },
    {
        "group": "6. Interpretability & error analysis",
        "title": "Step 32: Perform row-level error analysis",
        "instruction": (
            "Create error_analysis indexed like X_test and include "
            "actual, predicted, probability and correct columns. "
            "The correct column should indicate whether each "
            "prediction matches the actual class."
        ),
        "placeholder": (
            "error_analysis = pd.DataFrame(\n"
            "    {\n"
            '        "actual": y_test,\n'
            '        "predicted": test_predictions,\n'
            '        "probability": test_probabilities,\n'
            "    },\n"
            "    index=X_test.index,\n"
            ")\n\n"
            'error_analysis["correct"] = (\n'
            '    error_analysis["actual"]\n'
            '    == error_analysis["predicted"]\n'
            ")\n\n"
            "display(\n"
            "    error_analysis[\n"
            '        ~error_analysis["correct"]\n'
            "    ]\n"
            ")"
        ),
        "validator": validate_error_analysis,
        "input_height": "330px",
    },
    {
        "group": "7. Packaging & inference",
        "title": "Step 33: Create model metadata",
        "instruction": (
            "Create model_metadata documenting the model type, "
            "UTC creation time, reproducibility settings, selected "
            "parameters, validation score, holdout metrics, "
            "threshold, selected features, input feature columns "
            "and target names."
        ),
        "placeholder": (
            "model_metadata = {\n"
            '    "model_type": (\n'
            '        "LogisticRegression pipeline"\n'
            "    ),\n"
            '    "created_at_utc": (\n'
            "        datetime.now(timezone.utc)\n"
            "        .isoformat()\n"
            "    ),\n"
            '    "random_state": RANDOM_STATE,\n'
            '    "test_size": TEST_SIZE,\n'
            '    "cross_validation_folds": N_SPLITS,\n'
            '    "best_parameters": (\n'
            "        grid_search.best_params_\n"
            "    ),\n"
            '    "cross_validation_roc_auc": float(\n'
            "        grid_search.best_score_\n"
            "    ),\n"
            '    "holdout_metrics": (\n'
            "        holdout_metrics.to_dict()\n"
            "    ),\n"
            '    "decision_threshold": (\n'
            "        decision_threshold\n"
            "    ),\n"
            '    "selected_features": selected_features,\n'
            '    "feature_columns": X.columns.tolist(),\n'
            '    "target_names": (\n'
            "        cancer.target_names.tolist()\n"
            "    ),\n"
            "}\n\n"
            "print(\n"
            "    json.dumps(\n"
            "        model_metadata,\n"
            "        indent=2,\n"
            "    )\n"
            ")"
        ),
        "validator": validate_model_metadata,
        "input_height": "480px",
    },
    {
        "group": "7. Packaging & inference",
        "title": "Step 34: Package and save the model",
        "instruction": (
            "Create model_bundle containing final_model, "
            "decision_threshold and model_metadata. Save it with "
            "joblib to breast_cancer_pipeline.joblib and store the "
            "path in artifact_path."
        ),
        "placeholder": (
            "model_bundle = {\n"
            '    "model": final_model,\n'
            '    "threshold": decision_threshold,\n'
            '    "metadata": model_metadata,\n'
            "}\n\n"
            "artifact_path = Path(\n"
            '    "breast_cancer_pipeline.joblib"\n'
            ")\n\n"
            "joblib.dump(\n"
            "    model_bundle,\n"
            "    artifact_path,\n"
            ")\n\n"
            "print(artifact_path.resolve())"
        ),
        "validator": validate_model_bundle,
        "input_height": "300px",
    },
    {
        "group": "7. Packaging & inference",
        "title": "Step 35: Create a reusable prediction function",
        "instruction": (
            "Define predict_with_bundle(bundle, new_data). It "
            "should validate the input columns, calculate positive "
            "class probabilities, apply the saved threshold and "
            "return a DataFrame containing prediction, "
            "prediction_name, probability and threshold."
        ),
        "placeholder": (
            "def predict_with_bundle(bundle, new_data):\n"
            "    model = bundle[\"model\"]\n"
            "    threshold = bundle[\"threshold\"]\n"
            "    metadata = bundle[\"metadata\"]\n\n"
            "    expected_columns = metadata[\n"
            '        "feature_columns"\n'
            "    ]\n\n"
            "    missing_columns = set(\n"
            "        expected_columns\n"
            "    ) - set(new_data.columns)\n\n"
            "    if missing_columns:\n"
            "        raise ValueError(\n"
            "            \"Missing required columns: \"\n"
            "            f\"{sorted(missing_columns)}\"\n"
            "        )\n\n"
            "    ordered_data = new_data[\n"
            "        expected_columns\n"
            "    ]\n\n"
            "    probabilities = (\n"
            "        model\n"
            "        .predict_proba(ordered_data)\n"
            "        [:, 1]\n"
            "    )\n\n"
            "    predictions = (\n"
            "        probabilities >= threshold\n"
            "    ).astype(int)\n\n"
            "    target_names = np.asarray(\n"
            "        metadata[\"target_names\"]\n"
            "    )\n\n"
            "    return pd.DataFrame(\n"
            "        {\n"
            '            "prediction": predictions,\n'
            '            "prediction_name": (\n'
            "                target_names[predictions]\n"
            "            ),\n"
            '            "probability": probabilities,\n'
            '            "threshold": threshold,\n'
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
        "input_height": "620px",
    },
]


def create_activity():
    return GuidedActivity(
        title=(
            "Professional classification workflow"
        ),
        introduction=(
            "Develop a reproducible, leakage-safe machine-learning "
            "classification workflow. The activity covers data "
            "quality checks, a protected holdout set, baseline "
            "comparison, pipeline construction, feature selection, "
            "stratified cross-validation, hyperparameter tuning, "
            "out-of-fold threshold optimisation, final evaluation, "
            "interpretability, error analysis, metadata capture, "
            "model packaging and reusable inference."
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
            "cancer": cancer,
            "classification_data": classification_data,
            "RANDOM_STATE": RANDOM_STATE,
            "TEST_SIZE": TEST_SIZE,
            "N_SPLITS": N_SPLITS,
            "DummyClassifier": DummyClassifier,
            "StandardScaler": StandardScaler,
            "SelectKBest": SelectKBest,
            "f_classif": f_classif,
            "LogisticRegression": LogisticRegression,
            "Pipeline": Pipeline,
            "train_test_split": train_test_split,
            "StratifiedKFold": StratifiedKFold,
            "cross_val_score": cross_val_score,
            "cross_val_predict": cross_val_predict,
            "GridSearchCV": GridSearchCV,
            "accuracy_score": accuracy_score,
            "balanced_accuracy_score": (
                balanced_accuracy_score
            ),
            "precision_score": precision_score,
            "recall_score": recall_score,
            "f1_score": f1_score,
            "roc_auc_score": roc_auc_score,
            "precision_recall_curve": (
                precision_recall_curve
            ),
            "confusion_matrix": confusion_matrix,
            "classification_report": (
                classification_report
            ),
            "permutation_importance": (
                permutation_importance
            ),
        },
        preview=show_dataset,
    )