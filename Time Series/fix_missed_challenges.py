import json

with open("M8.2_TimeSeriesAnalysis_learner_.ipynb", "r") as f:
    nb = json.load(f)

new_cells = []

def create_template_cell(code):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "<p><b>Code Template:</b></p>\n",
            f"<pre><code>{code}</code></pre>"
        ]
    }

def create_code_cell(code=""):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [code]
    }

def create_hint_cell(hint):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "<details>\n",
            "<summary>💡 Click here for a hint</summary>\n",
            "<br>\n",
            f"{hint}\n",
            "</details>"
        ]
    }

i = 0
while i < len(nb['cells']):
    cell = nb['cells'][i]
    source = "".join(cell.get('source', []))
    
    if i == 27 or i == 29:
        if i == 27:
            new_cells.append(create_template_cell(
"""# Apply multiplicative decomposition
sdc_m = seasonal_decompose(..., model='...')

# Create multiplicative model
multiplicative_model = sdc_m.trend ... sdc_m.seasonal

# Plot against original time series
plt.plot(ts['unemployment_claim'], label='Original', color='grey')
plt.plot(multiplicative_model, label='Multiplicative', linestyle='--', color='red')
plt.legend()"""))
            new_cells.append(create_code_cell())
            new_cells.append(create_hint_cell("Set <code>model='multiplicative'</code> or <code>'m'</code> in <code>seasonal_decompose()</code>. Multiply (<code>*</code>) the trend and seasonal components instead of adding them."))
        # i == 29 we just skip appending because it's handled above
    else:
        new_cells.append(cell)
    i += 1

nb['cells'] = new_cells

with open("M8.2_TimeSeriesAnalysis_learner_.ipynb", "w") as f:
    json.dump(nb, f, indent=1)

