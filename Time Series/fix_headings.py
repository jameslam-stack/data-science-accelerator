import json
import re

with open("M8.2_TimeSeriesAnalysis_learner_.ipynb", "r") as f:
    nb = json.load(f)

def sentence_case(text):
    # Split the heading into hashes and the rest
    match = re.match(r'^(#+)\s*(.*)', text)
    if not match:
        return text
    hashes, title = match.groups()
    if not title:
        return text
    
    # Check if the title starts with "Challenge:" or "Stretch:" or "Optional stretch:"
    prefix_match = re.match(r'^(Challenge:|Stretch:|Optional stretch[:]?|Practice:|Review:|AutoML for time series:)(.*)', title, re.IGNORECASE)
    
    if prefix_match:
        prefix, rest = prefix_match.groups()
        rest = rest.strip()
        if rest:
            rest = rest[0].upper() + rest[1:].lower()
        title = prefix.capitalize() + ' ' + rest if rest else prefix.capitalize()
    else:
        title = title[0].upper() + title[1:].lower()
    
    return f"{hashes} {title}"

for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'markdown':
        source = cell.get('source', [])
        new_source = []
        for line in source:
            if line.startswith('#'):
                # Handle cases where heading is mixed with other things but generally lines starting with # are headings
                # However, avoid changing things inside html or code blocks. We assume # at start of line is a heading.
                # Actually, some code templates have # which are comments! They shouldn't be touched in markdown unless they are proper headers.
                # Wait, code templates are inside <pre><code> so they don't start with # at the beginning of the string, or do they?
                # The line might start with # inside a markdown cell. 
                # Let's only change if the cell does not contain '<p><b>Code Template:</b></p>'
                if '<p><b>Code Template:</b></p>' not in "".join(source):
                    line = sentence_case(line)
            new_source.append(line)
        cell['source'] = new_source

with open("M8.2_TimeSeriesAnalysis_learner_.ipynb", "w") as f:
    json.dump(nb, f, indent=1)

