import json

with open("M8.2_TimeSeriesAnalysis_learner_.ipynb", "r") as f:
    nb = json.load(f)

for i, cell in enumerate(nb['cells']):
    source = "".join(cell.get('source', []))
    print(f"Cell {i} ({cell['cell_type']}): {source[:100].strip()}...")
