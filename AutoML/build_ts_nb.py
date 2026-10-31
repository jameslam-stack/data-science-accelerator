import json

cells = []

def md(source):
    return {"cell_type": "markdown", "metadata": {}, "source": source}

def code(source):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": source}

# Title
cells.append(md([
    '<img src=https://courseware.decoded.com/images/decoded/logo-decoded.png align=left width=100px/>',
    '',
    '# M09 Time Series Forecasting with PyCaret',
    '---',
    '<small>© 2026 Decoded Limited. All rights reserved. Website: https://decoded.com</small>',
]))

# Welcome
cells.append(md([
    '## Welcome to the Time Series Forecasting Workflow',
    '',
    'In this notebook, you will build a complete **time series forecasting** pipeline using PyCaret. We will focus on the **monthly sales dataset** and demonstrate how to:',
    '',
    '| Step | What you do |',
    '| ---- | ------------ |',
    '| 1. Load & Explore | Load time series data and perform quick sanity checks |',
    '| 2. Feature Engineering | Create lag, rolling, and calendar features |',
    '| 3. PyCaret Setup | Initialise the time series environment with `setup()` |',
    '| 4. Compare Models | Test multiple time series models (RandomForest, XGBoost, etc.) |',
    '| 5. Tune Best Model | Hyperparameter optimisation with grid search |',
    '| 6. Evaluate | Performance metrics on holdout data |',
    '| 7. Deploy | Save the pipeline for future predictions |',
    '',
    '### Key Features',
    '- **TimeSeriesSplit**: Proper chronological train/test splitting',
    '- **Feature Engineering**: Automatic lag, rolling, and calendar features',
    '- **Multiple Models**: Compare tree-based, linear, and ensemble methods',
    '- **Reusable Pipeline**: Save everything for production deployment',
    '',
    '### Dataset',
    'Monthly sales data with 96 observations (2018-2024) featuring:',
    '- **Date**: Monthly time index',
    '- **Sales**: Target variable to forecast',
    '- **Promotion**: Binary promotional indicators',
    '- **Calendar Features**: Month, season, etc.',
]))

# Install + imports
cells.append(md(['## 0. Install and import libraries']))
cells.append(code([
    '# Install dependencies (uncomment if running for the first time)',
    '# !pip install pycaret ydata-profiling scikit-learn pandas numpy',
    '',
    '%matplotlib inline',
    'import numpy as np',
    'import pandas as pd',
    'import matplotlib.pyplot as plt',
    'import seaborn as sns',
    'from pycaret.datasets import get_data',
    'from pycaret.time_series import *',
    'from sklearn.model_selection import train_test_split',
    'from datetime import datetime, timezone',
    '',
    'print("All libraries imported successfully for time series forecasting.")'
]))

# Load data
cells.append(md(['## 1. Load and explore time series data']))
cells.append(code([
    '# Load the monthly sales dataset',
    'ts_data = get_data("time_series", verbose=False)',
    '',
    'print("Dataset shape:", ts_data.shape)',
    'print("\\nFirst 10 rows:")',
    'print(ts_data.head(10))',
    '',
    'print("\\nTarget variable info:")',
    'print(f"Sales - Mean: {ts_data[\'sales\'].mean():.2f}, Std: {ts_data[\'sales\'].std():.2f}")',
    'print(f"Sales - Min: {ts_data[\'sales\'].min():.2f}, Max: {ts_data[\'sales\'].max():.2f}")',
    '',
    'print("\\nDate range:", ts_data[\'date\'].min(), "to", ts_data[\'date\'].max())',
    '',
    'ts_data[\'date\'] = pd.to_datetime(ts_data[\'date\'])',
    'ts_data.set_index(\'date\', inplace=True)',
    'ts_data.sort_index(inplace=True)'
]))

# Quick EDA
cells.append(md(['## 2. Quick Exploratory Analysis']))
cells.append(code([
    '# Basic time series plots',
    'fig, axes = plt.subplots(2, 2, figsize=(15, 10))',
    '',
    '# Sales over time',
    'axes[0, 0].plot(ts_data.index, ts_data[\'sales\'], linewidth=2)',
    'axes[0, 0].set_title("Monthly Sales Over Time")',
    'axes[0, 0].set_ylabel("Sales")',
    '',
    '# Sales with promotions',
    'axes[0, 1].plot(ts_data.index, ts_data[\'sales\'], label=\'Sales\')',
    'axes[0, 1].plot(ts_data.index, ts_data[\'promotion\'] * 20, label=\'Promotion (scaled)\', color=\'red\')',
    'axes[0, 1].set_title("Sales vs Promotions")',
    'axes[0, 1].legend()',
    '',
    '# Sales distribution',
    'axes[1, 0].hist(ts_data[\'sales\'], bins=20, alpha=0.7, edgecolor=\'black\')',
    'axes[1, 0].set_title("Sales Distribution")',
    'axes[1, 0].set_xlabel("Sales")',
    '',
    '# Rolling statistics',
    'rolling_mean = ts_data[\'sales\'].rolling(window=12).mean()',
    'axes[1, 1].plot(ts_data.index, ts_data[\'sales\'], label=\'Original\', alpha=0.5)',
    'axes[1, 1].plot(ts_data.index, rolling_mean, label=\'12-month Rolling Mean\', linewidth=2)',
    'axes[1, 1].set_title("Sales with 12-Month Rolling Mean")',
    'axes[1, 1].legend()',
    '',
    'plt.tight_layout()',
    'plt.show()',
    '',
    '# Check for missing values',
    'print("Missing values:", ts_data.isnull().sum().sum())',
    'print("\\nDate frequency:", ts_data.index.freq or "Monthly data")'
]))

# Feature engineering preview
cells.append(md(['## 3. Feature Engineering Preview']))
cells.append(code([
    '# Create a sample of engineered features (in real workflow, PyCaret does this automatically)',
    'ts_features = ts_data.copy()',
    '',
    '# Calendar features',
    'ts_features[\'month\'] = ts_features.index.month',
    'ts_features[\'month_sin\'] = np.sin(2 * np.pi * ts_features[\'month\'] / 12)',
    'ts_features[\'month_cos\'] = np.cos(2 * np.pi * ts_features[\'month\'] / 12)',
    '',
    '# Lag features',
    'for lag in [1, 2, 3, 12]:',
    '    ts_features[f\'lag_{lag}\'] = ts_features[\'sales\'].shift(lag)',
    '',
    '# Rolling features',
    'ts_features[\'rolling_mean_3\'] = ts_features[\'sales\'].rolling(window=3).mean()',
    'ts_features[\'rolling_mean_6\'] = ts_features[\'sales\'].rolling(window=6).mean()',
    '',
    '# Drop rows with NaN values from lag/rolling features',
    'ts_features = ts_features.dropna()',
    '',
    'print(f"Engineered features shape: {ts_features.shape}")',
    'print("\\nFeature columns:", list(ts_features.columns))',
    '',
    'print("\\nSample engineered data:")',
    'print(ts_features[[\'sales\', \'month\', \'month_sin\', \'month_cos\', \'lag_1\', \'lag_12\', \'rolling_mean_3\']].head())'
]))

# PyCaret time series setup
cells.append(md(['## 4. PyCaret Time Series Setup']))
cells.append(code([
    '# Step 1: Setup the time series environment',
    'ts_setup = setup(',
    '    data=ts_data,',
    '    target=\'sales\',',
    '    session_id=123,',
    '    fold=5,',
    '    n_jobs=1,',
    '    verbose=False',
    ')',
    '',
    'print("Time series environment setup complete!")',
    'print("Features:", ts_setup)'
]))

# Model comparison
cells.append(md(['## 5. Compare Time Series Models']))
cells.append(code([
    '# Step 2: Compare different time series models',
    '# Note: PyCaret automatically handles TimeSeriesSplit for each model',
    'ts_models = compare_models(sort="RMSE", budget_time=2.0)',
    '',
    'print("\\nModel comparison completed!")',
    'print("Best model:", type(ts_models).__name__)',
    '',
    '# You can also specify which models to compare',
    'selected_models = ["RandomForest", "XGBoost", "LightGBM"]',
    'ts_models_selected = compare_models(',
    '    model_to_test=selected_models,',
    '    sort="RMSE",',
    '    budget_time=2.0',
    ')',
    '',
    'print("\\nSelected models comparison:")',
    'print(type(ts_models_selected).__name__)'
]))

# Hyperparameter tuning
cells.append(md(['## 6. Hyperparameter Tuning']))
cells.append(code([
    '# Step 3: Tune the best model',
    'ts_tuned = tune_model(ts_models, optimize="RMSE")',
    '',
    'print("\\nModel tuning completed!")',
    'print("Tuned model:", type(ts_tuned).__name__)',
    '',
    '# Step 4: Finalize on entire dataset',
    'ts_final = finalize_model(ts_tuned)',
    '',
    'print("Final model prepared for deployment:", type(ts_final).__name__)'
]))

# Evaluation and visualization
cells.append(md(['## 7. Evaluation and Visualization']))
cells.append(code([
    '# Step 5: Generate predictions (PyCaret uses TimeSeriesSplit internally)',
    'ts_predictions = predict_model(ts_final)',
    '',
    'print("\\nPredictions generated!")',
    'print("Prediction columns:", ts_predictions.columns.tolist())',
    '',
    '# Show sample predictions',
    'print("\\nSample predictions:")',
    'display(ts_predictions[[\'sales\', \'prediction_label\', \'prediction_score\']].head(10))',
    '',
    '# Step 6: Visualise performance',
    'print("\\nGenerating time series plots...")',
    'plot_model(ts_tuned, plot="time_series")',
    'plot_model(ts_tuned, plot="residuals")',
    'plot_model(ts_tuned, plot="error")',
    '',
    '# Forecast next periods',
    'print("\\nGenerating forecasts for next 12 months...")',
    'forecast_next = predict_model(ts_final, periods=12)',
    'print("\\n12-Month Forecast:")',
    'display(forecast_next[[\'prediction_label\']])',
    'print("\\nForecast confidence intervals:")',
    'display(forecast_next[["prediction_label", "prediction_score"]])'
]))

# Save and deploy
cells.append(md(['## 8. Save and Deploy']))
cells.append(code([
    '# Step 7: Save the time series pipeline',
    'save_model(ts_tuned, "time_series_forecast_pipeline")',
    'print("\\nTime series model saved as \'time_series_forecast_pipeline.pkl\'")',
    '',
    '# Step 8: Load and verify',
    'loaded_ts = load_model("time_series_forecast_pipeline")',
    'print("\\nLoaded model type:", type(loaded_ts).__name__)',
    '',
    '# Generate new predictions with loaded model',
    'new_predictions = predict_model(loaded_ts, data=ts_data)',
    'print("\\nSample predictions with loaded model:")',
    'display(new_predictions[[\'sales\', \'prediction_label\']].head(5))'
]))

# Summary
cells.append(md([
    '---',
    '## Summary',
    '',
    'You have completed the full time series forecasting workflow with PyCaret:',
    '',
    '| Step | What you did |',
    '| ---- | ------------ |',
    '| 1. Load & Explore | Loaded monthly sales data and plotted trends |',
    '| 2. Feature Engineering | Created lag, rolling, and calendar features |',
    '| 3. PyCaret Setup | Initialised time series environment with proper validation |',
    '| 4. Compare Models | Tested multiple time series algorithms automatically |',
    '| 5. Tune Best Model | Optimised hyperparameters for the best performer |',
    '| 6. Evaluate | Generated predictions and visualised performance |',
    '| 7. Deploy | Saved the pipeline and verified with load_model() |',
    '',
    '### Key Time Series Takeaways',
    '',
    '**Time Series Splits:**',
    '- Used `TimeSeriesSplit` instead of random splits',
    '- Preserves temporal order for realistic evaluation',
    '- 5-fold cross-validation across time periods',
    '',
    '**PyCaret Advantages:**',
    '- Handles feature engineering automatically',
    '- Built-in time series cross-validation',
    '- Multiple model comparison with a single command',
    '- Hyperparameter optimisation with grid search',
    '',
    '**Production Ready:**',
    '- Saved pipeline includes preprocessing and model',
    '- Can predict on raw time series data later',
    '- Reproducible with `session_id` for consistency',
    '',
    '### Next Steps',
    '- Use the saved pipeline in production for sales forecasting',
    '- Monitor model performance on new data',
    '- Consider ensemble methods for better accuracy',
    '- Implement automated retraining pipelines',
]))

# Footer
cells.append(md([
    '---',
    '<small>© 2026 Decoded Limited. All rights reserved.</small>',
]))

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.12.0"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 5
}

out = "/Users/james.lam/Workshops/Data Science Accelerator/AutoML/M7.4_Time_Series_Forecasting_Learner.ipynb"
with open(out, "w") as f:
    json.dump(nb, f, indent=1)
    f.write("\n")
print("Wrote", out, "with", len(cells), "cells")
