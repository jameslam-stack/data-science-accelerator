import json

with open("M8.1_Time_Series_Pandas_learner.ipynb", "r") as f:
    nb = json.load(f)

for i, cell in enumerate(nb['cells']):
    source = "".join(cell.get('source', []))
    if '<pre>' in source or '<details>' in source or 'FILL IN' in source:
        print(f"Cell {i} ({cell['cell_type']}): {source}\n---\n")
