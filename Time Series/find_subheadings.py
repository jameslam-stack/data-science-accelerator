import json

with open("M8.2_TimeSeriesAnalysis_learner_.ipynb", "r") as f:
    nb = json.load(f)

for i in range(57, 65):
    if nb['cells'][i]['cell_type'] == 'markdown':
        print(f"Cell {i}:\n{''.join(nb['cells'][i].get('source', []))}\n---")
