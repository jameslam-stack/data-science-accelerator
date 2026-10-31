import json
import re

with open("M8.2_TimeSeriesAnalysis_learner_.ipynb", "r") as f:
    nb = json.load(f)

def replace_us_to_uk(text):
    text = re.sub(r'\bvisualize\b', 'visualise', text, flags=re.IGNORECASE)
    text = re.sub(r'\bvisualizing\b', 'visualising', text, flags=re.IGNORECASE)
    text = re.sub(r'\bcolor\b', 'colour', text, flags=re.IGNORECASE)
    text = re.sub(r'\bpractice\b', 'practise', text, flags=re.IGNORECASE) # noun vs verb is tricky but 'practise' is used as verb in UK. The prompt says 'leave blank cells for learners to practise'.
    return text

for cell in nb['cells']:
    if cell['cell_type'] in ['markdown', 'code']:
        source = cell.get('source', [])
        new_source = []
        for line in source:
            new_source.append(replace_us_to_uk(line))
        cell['source'] = new_source

with open("M8.2_TimeSeriesAnalysis_learner_.ipynb", "w") as f:
    json.dump(nb, f, indent=1)

