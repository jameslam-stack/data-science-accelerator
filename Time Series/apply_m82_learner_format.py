import json

with open("M8.2_TimeSeriesAnalysis_learner_.ipynb", "r") as f:
    nb = json.load(f)

new_cells = []

# define helper function to create cells
def create_template_cell(code):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "<p><b>Code Template:</b></p>\n",
            f"<pre><code>{code}</code></pre>"
        ]
    }

def create_code_cell(code=""):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [code]
    }

def create_hint_cell(hint):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "<details>\n",
            "<summary>💡 Click here for a hint</summary>\n",
            "<br>\n",
            f"{hint}\n",
            "</details>"
        ]
    }

i = 0
while i < len(nb['cells']):
    cell = nb['cells'][i]
    source = "".join(cell.get('source', []))
    
    # Check if this cell is a challenge text that expects a code following
    # Markdown Cell 32 -> Code Cell 33
    if i == 33: # Code Cell 33: level = ...
        new_cells.append(create_template_cell(
"""# Plot the level of the series with a rolling window to check for stationarity
ts['unemployment_claim'].plot(marker='o')
level = ts['unemployment_claim'].rolling(window=13, center=True).mean()
level.plot(linestyle='--',label='level', color='red')
plt.legend()"""))
        new_cells.append(create_code_cell())
        new_cells.append(create_hint_cell("Use <code>.rolling(window=13)</code> to calculate the level which smooths out the seasonality, then chain with <code>.mean()</code> and <code>.plot()</code>."))
    elif i == 47: # empty code cell for challenge 2
        new_cells.append(create_template_cell(
"""# Run the ADF test again to check for stationarity
result = adfuller(...)
print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])
"""))
        new_cells.append(create_code_cell())
        new_cells.append(create_hint_cell("Pass your differenced time series (<code>ts_diff</code>) to the <code>adfuller()</code> function from <code>statsmodels</code>."))
    elif i == 49: # empty code cell for ACF plot
        new_cells.append(create_template_cell(
"""# Run an ACF plot again to check for seasonality
plot_acf(..., lags=30)
plt.show()"""))
        new_cells.append(create_code_cell())
        new_cells.append(create_hint_cell("Use <code>plot_acf()</code> and pass in your differenced time series to check for remaining seasonality."))
    elif i == 56: # Train test split
        new_cells.append(create_template_cell(
"""threshold = int(len(ts) * 0.8)
train = ts['unemployment_claim'][:threshold]
test = ts['unemployment_claim'][threshold:]"""))
        new_cells.append(create_code_cell())
        new_cells.append(create_hint_cell("Use list slicing notation (e.g., <code>[:threshold]</code> and <code>[threshold:]</code>) to separate the data into train and test sets."))
    elif i == 73: # visualising predictions
        new_cells.append(create_template_cell(
"""# Visualise the predictions and confidence intervals
plt.plot(train, label="train", linestyle='--')
plt.plot(test, label="test", color="navy", linestyle='--')
plt.plot(autoarima_preds, label="AutoArima predictions", marker='x', linestyle='--')
plt.fill_between(test.index, cf[0], cf[1], alpha=.3, color='red') 
plt.legend(loc='upper left')"""))
        new_cells.append(create_code_cell())
        new_cells.append(create_hint_cell("Use <code>plt.fill_between()</code> to shade the area representing confidence intervals."))
    elif i == 83: # lagged features
        new_cells.append(create_template_cell(
"""# Create lagged 1 and 2 as features 
data['claim_lag1'] = data['unemployment_claim'].shift(1)
data['claim_lag2'] = ..."""))
        new_cells.append(create_code_cell())
        new_cells.append(create_hint_cell("Use the <code>.shift()</code> method on your time series column, passing the number of periods to shift (e.g. <code>2</code>) to create lag features."))
    elif i == 87: # Train test split Random Forest
        new_cells.append(create_template_cell(
"""# Train-test split (use last 26 weeks for testing)
train_size = threshold
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = ..."""))
        new_cells.append(create_code_cell())
        new_cells.append(create_hint_cell("Use the same index slicing (<code>[:train_size]</code> and <code>[train_size:]</code>) for the target variable <code>y</code> as you did for the features <code>X</code>."))
    elif i == 90: # Make predictions for training set
        new_cells.append(create_template_cell(
"""# Make predictions on the training set and calculate metrics
rf_predictions_train = rf_model.predict(X_train)

# Make predictions for the test set
rf_predictions = rf_model.predict(X_test)

# Calculate metrics for the test set
rf_rmse = np.sqrt(mean_squared_error(y_test, rf_predictions))
print(f'Random Forest RMSE: {rf_rmse:.2f}')"""))
        new_cells.append(create_code_cell())
        new_cells.append(create_hint_cell("Use <code>.predict()</code> on your fitted model passing the test features, then calculate RMSE using <code>np.sqrt(mean_squared_error())</code>."))
    elif i == 95: # evaluate gradient boosting
        new_cells.append(create_template_cell(
"""# Make predictions on the training set and calculate metrics
gradient_model_predictions_train = gradient_model.predict(X_train)

# Calculate metrics for the training set
gradient_model_rmse_train = np.sqrt(mean_squared_error(y_train, gradient_model_predictions_train))
"""))
        new_cells.append(create_code_cell())
        new_cells.append(create_hint_cell("Use <code>.predict()</code> with the training data (<code>X_train</code>) and evaluate using MSE, MAE, or RMSE."))
    elif i == 96: # evaluate gradient boosting test
        new_cells.append(create_template_cell(
"""# Make predictions for the test set
gradient_model_predictions = gradient_model.predict(X_test)"""))
        new_cells.append(create_code_cell())
        new_cells.append(create_hint_cell("Use <code>.predict()</code> on the fitted <code>gradient_model</code> passing the testing features (<code>X_test</code>)."))
    elif i == 97: # evaluate gradient boosting metrics
        new_cells.append(create_template_cell(
"""# Calculate metrics for the test set
gradient_model_rmse = np.sqrt(mean_squared_error(y_test, gradient_model_predictions))
print(f'Gradient Boosting RMSE: {gradient_model_rmse:.2f}')"""))
        new_cells.append(create_code_cell())
        new_cells.append(create_hint_cell("Pass the actual values (<code>y_test</code>) and predicted values to <code>mean_squared_error()</code>, then use <code>np.sqrt()</code>."))
    else:
        new_cells.append(cell)
    i += 1

nb['cells'] = new_cells

with open("M8.2_TimeSeriesAnalysis_learner_.ipynb", "w") as f:
    json.dump(nb, f, indent=1)

