# Ex.No: 1B                     CONVERSION OF NON STATIONARY TO STATIONARY DATA


### AIM:
To perform regular differncing,seasonal adjustment and log transformatio on international airline passenger data
### ALGORITHM:
1. Import the required packages like pandas and numpy
2. Read the data using the pandas
3. Perform the data preprocessing if needed and apply regular differncing,seasonal adjustment,log transformation.
4. Plot the data according to need, before and after regular differncing,seasonal adjustment,log transformation.
5. Display the overall results.
### PROGRAM:
```python
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
data=pd.read_csv("/content/rainfall.csv")
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose

# Load dataset
data = pd.read_csv("/content/rainfall.csv")

# Detect date column
date_col = None
for col in data.columns:
    if 'date' in col.lower() or 'month' in col.lower():
        date_col = col
        break

if date_col is None:
    raise ValueError("No date column found. Please check your CSV headers.")

# Convert to datetime safely
data[date_col] = pd.to_datetime(data[date_col], errors='coerce')
data = data.dropna(subset=[date_col])

# Set index
data.set_index(date_col, inplace=True)

# Ensure numeric columns
for col in ['rainfall', 'temperature', 'humidity', 'wind_speed']:
    if col in data.columns:
        data[col] = pd.to_numeric(data[col], errors='coerce')

# Drop rows with NaNs in rainfall
data = data.dropna(subset=['rainfall'])

# ---------------------------
# 📊 OVERALL VIEW
# ---------------------------
print("==== Overall Summary ====")
print(data[['rainfall', 'temperature', 'humidity', 'wind_speed']].describe())

plt.figure(figsize=(14, 8))
plt.plot(data.index, data['rainfall'], label='Rainfall (mm)')
plt.plot(data.index, data['temperature'], label='Temperature (°C)')
plt.plot(data.index, data['humidity'], label='Humidity (%)')
plt.plot(data.index, data['wind_speed'], label='Wind Speed')
plt.title("Overall Weather Trends")
plt.xlabel("Date")
plt.ylabel("Values")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# ---------------------------
# 🔍 RAINFALL DECOMPOSITION
# ---------------------------

# Differencing
data['rainfall_diff'] = data['rainfall'] - data['rainfall'].shift(1)

# Seasonal decomposition
result = seasonal_decompose(data['rainfall'].dropna(), model='additive', period=7)
data['rainfall_sea_diff'] = result.resid

# Log transform
data['rainfall_log'] = np.log(data['rainfall'] + 1)
data['rainfall_log_diff'] = data['rainfall_log'] - data['rainfall_log'].shift(1)

# Seasonal decomposition on log-diff
result = seasonal_decompose(data['rainfall_log_diff'].dropna(), model='additive', period=7)
data['rainfall_log_seasonal_diff'] = result.resid

# Plot decomposition steps
plt.figure(figsize=(16, 16))

plt.subplot(6, 1, 1)
plt.plot(data['rainfall'], label='Original')
plt.legend(loc='best')
plt.title('Original Rainfall Data')
plt.xlabel('Date')
plt.ylabel('Rainfall (mm)')

plt.subplot(6, 1, 2)
plt.plot(data['rainfall_diff'], label='Regular Difference')
plt.legend(loc='best')
plt.title('Regular Differencing')
plt.xlabel('Date')
plt.ylabel('Differenced Rainfall')

plt.subplot(6, 1, 3)
plt.plot(data['rainfall_sea_diff'], label='Seasonal Adjustment')
plt.legend(loc='best')
plt.title('Seasonal Adjustment')
plt.xlabel('Date')
plt.ylabel('Seasonally Adjusted Rainfall')

plt.subplot(6, 1, 4)
plt.plot(data['rainfall_log'], label='Log Transformation')
plt.legend(loc='best')
plt.title('Log Transformation')
plt.xlabel('Date')
plt.ylabel('Log(Rainfall)')

plt.subplot(6, 1, 5)
plt.plot(data['rainfall_log_diff'], label='Log Transformation + Regular Differencing')
plt.legend(loc='best')
plt.title('Log Transformation + Regular Differencing')
plt.xlabel('Date')
plt.ylabel('Rdiff(Log(Rainfall))')

plt.subplot(6, 1, 6)
plt.plot(data['rainfall_log_seasonal_diff'], label='Log + Diff + Seasonal Adjustment')
plt.legend(loc='best')
plt.title('Log + Regular Differencing + Seasonal Differencing')
plt.xlabel('Date')
plt.ylabel('Sdiff(Rdiff(Log(Rainfall)))')

plt.tight_layout()
plt.show()


```


### OUTPUT:


REGULAR DIFFERENCING:

<img width="2600" height="442" alt="Screenshot 2025-08-25 222041" src="https://github.com/user-attachments/assets/fab8b794-3539-4288-b57a-c85ccf18b190" />


SEASONAL ADJUSTMENT:
<img width="2617" height="448" alt="Screenshot 2025-08-25 222055" src="https://github.com/user-attachments/assets/29aef9de-643d-4d46-8ab7-8212e8b7c203" />


LOG TRANSFORMATION:
<img width="2607" height="437" alt="Screenshot 2025-08-25 222105" src="https://github.com/user-attachments/assets/f41033be-e01b-4ea5-a3f4-7481b38b0908" />



### RESULT:
Thus we have created the python code for the conversion of non stationary to stationary data on international airline passenger
data.
