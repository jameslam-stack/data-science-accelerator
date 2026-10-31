import json

with open("M8.2_TimeSeriesAnalysis_learner_.ipynb", "r") as f:
    nb = json.load(f)

for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'markdown':
        source = "".join(cell.get('source', []))
        if 'Time series forecasting - training the model' in source:
            print(f"Cell {i}:\n{source}")
