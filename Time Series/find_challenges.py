import json

with open("M8.2_TimeSeriesAnalysis_learner_.ipynb", "r") as f:
    nb = json.load(f)

for i, cell in enumerate(nb['cells']):
    source = "".join(cell.get('source', []))
    if cell['cell_type'] == 'code':
        if 'FILL IN' in source or 'Challenge' in source or '?' in source or source.strip() == '' or '#' in source:
            print(f"Code Cell {i}:\n{source}\n---\n")
    if 'Challenge' in source:
        print(f"Markdown Cell {i}:\n{source}\n---\n")
