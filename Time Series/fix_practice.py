import json

with open("M8.2_TimeSeriesAnalysis_learner_.ipynb", "r") as f:
    nb = json.load(f)

for cell in nb['cells']:
    source = cell.get('source', [])
    new_source = []
    for line in source:
        line = line.replace('best practise', 'best practice')
        line = line.replace('Best practise', 'Best practice')
        new_source.append(line)
    cell['source'] = new_source

with open("M8.2_TimeSeriesAnalysis_learner_.ipynb", "w") as f:
    json.dump(nb, f, indent=1)

