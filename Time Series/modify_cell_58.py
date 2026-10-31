import json

with open("M8.2_TimeSeriesAnalysis_learner_.ipynb", "r") as f:
    nb = json.load(f)

source = nb['cells'][58]['source']
new_source = []
for line in source:
    line = line.replace('**Exponential Smoothing Models**', '[**Exponential Smoothing Models**](https://otexts.com/fpp3/expsmooth.html)')
    line = line.replace('**Moving Average Models**', '[**Moving Average Models**](https://otexts.com/fpp3/MA.html)')
    line = line.replace('**Autoregressive Models**', '[**Autoregressive Models**](https://otexts.com/fpp3/AR.html)')
    line = line.replace('**Box-Jenkins Models**', '[**Box-Jenkins Models**](https://otexts.com/fpp3/arima.html)')
    line = line.replace('**Deep Learning Models**', '[**Deep Learning Models**](https://machinelearningmastery.com/how-to-get-started-with-deep-learning-for-time-series-forecasting-7-day-mini-course/)')
    line = line.replace('**Tree-Based Models**', '[**Tree-Based Models**](https://machinelearningmastery.com/xgboost-for-time-series-forecasting/)')
    new_source.append(line)

nb['cells'][58]['source'] = new_source

with open("M8.2_TimeSeriesAnalysis_learner_.ipynb", "w") as f:
    json.dump(nb, f, indent=1)

