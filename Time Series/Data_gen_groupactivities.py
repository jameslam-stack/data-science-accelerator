import numpy as np
import pandas as pd

np.random.seed(42)

# Generate 2 years of daily dates
dates = pd.date_range(start="2024-01-01", end="2025-12-31", freq="D")
n = len(dates)

# Generate air quality signal
pm25 = (
    20
    + 8 * np.cos(2 * np.pi * np.arange(n) / 365)
    + np.random.normal(0, 3, n)
)

# Add bad sensor values (outliers)
pm25[100] = -15.0  # Impossible negative reading
pm25[450] = 300.0  # Sensor error spike

df = pd.DataFrame(
    {"reading_date": dates, "pm25_concentration": np.round(pm25, 2)}
)

# Drop 10% of rows to simulate missing log days
df = df.sample(frac=0.90, random_state=42).sort_values("reading_date")

# Mix date string formats
formatted_dates = []
for idx, dt in enumerate(df["reading_date"]):
    if idx % 3 == 0:
        formatted_dates.append(dt.strftime("%d/%m/%Y"))
    elif idx % 3 == 1:
        formatted_dates.append(dt.strftime("%Y-%m-%d"))
    else:
        formatted_dates.append(dt.strftime("%d-%b-%Y"))

df["reading_date"] = formatted_dates
df.to_csv("uk_air_quality.csv", index=False)
print("Dataset 'uk_air_quality.csv' created.")