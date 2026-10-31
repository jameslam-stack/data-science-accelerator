import json

with open("M8.2_TimeSeriesAnalysis_learner.ipynb", "r") as f:
    nb = json.load(f)

for i, cell in enumerate(nb['cells']):
    source = "".join(cell.get('source', []))
    if 'Challenge' in source or 'FILL IN' in source:
        print(f"Cell {i} ({cell['cell_type']}):\n{source[:100]}\n")
