import json

with open("M8.2_TimeSeriesAnalysis_learner_.ipynb", "r") as f:
    nb = json.load(f)

for i in range(len(nb['cells'])):
    cell = nb['cells'][i]
    if cell['cell_type'] == 'code' and not "".join(cell.get('source', [])).strip():
        # Check the previous cell to see what it's about
        if i > 0 and nb['cells'][i-1]['cell_type'] == 'markdown':
            prev_source = "".join(nb['cells'][i-1].get('source', []))
            if '<p><b>Code Template:</b></p>' in prev_source:
                # We found an empty code cell following a template!
                # Let's add a generic comment.
                if 'multiplicative' in prev_source:
                    cell['source'] = ['# Add your code here to apply multiplicative decomposition\n']
                elif 'level' in prev_source:
                    cell['source'] = ['# Add your code here to calculate and plot the level\n']
                elif 'ADF' in prev_source:
                    cell['source'] = ['# Add your code here to run the ADF test\n']
                elif 'ACF' in prev_source:
                    cell['source'] = ['# Add your code here to create the ACF plot\n']
                elif 'threshold' in prev_source:
                    cell['source'] = ['# Add your code here to create the train and test sets\n']
                elif 'predictions and confidence' in prev_source:
                    cell['source'] = ['# Add your code here to visualise the predictions and confidence intervals\n']
                elif 'claim_lag' in prev_source:
                    cell['source'] = ['# Add your code here to create the lag features\n']
                elif 'train_size' in prev_source:
                    cell['source'] = ['# Add your code here to split the features and target into train and test sets\n']
                elif 'rf_predictions_train' in prev_source:
                    cell['source'] = ['# Add your code here to make predictions and calculate RMSE for the Random Forest model\n']
                elif 'gradient_model_predictions_train' in prev_source:
                    cell['source'] = ['# Add your code here to evaluate the Gradient Boosting model on the training set\n']
                elif 'gradient_model_predictions =' in prev_source:
                    cell['source'] = ['# Add your code here to make predictions with the Gradient Boosting model on the test set\n']
                elif 'gradient_model_rmse =' in prev_source:
                    cell['source'] = ['# Add your code here to calculate the RMSE for the Gradient Boosting model\n']
                else:
                    cell['source'] = ['# Write your code here\n']

with open("M8.2_TimeSeriesAnalysis_learner_.ipynb", "w") as f:
    json.dump(nb, f, indent=1)
