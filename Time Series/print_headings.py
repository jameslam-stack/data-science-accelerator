import json
with open("M8.2_TimeSeriesAnalysis_learner_.ipynb", "r") as f:
    nb = json.load(f)

for i, cell in enumerate(nb['cells']):
    source = "".join(cell.get('source', []))
    if cell['cell_type'] == 'markdown':
        for line in source.split('\n'):
            if line.startswith('#'):
                print(f"Cell {i}: {line}")
