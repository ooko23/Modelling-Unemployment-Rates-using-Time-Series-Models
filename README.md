# Modelling-Unemployment-Rates-using-Time-Series-Models
## Project Overview
Using Time Series models such as ARIMA, SARIMA, and HOLT WINTERS models to model and forecast unemployment rates.
## Data Source
- Kenya's monthly unemployment data 
## Tools
- Python

## Data Cleaning
The data is clean for analysis

## Exploratory Data Analysis
- Descriptive statistics
- Trend chart
- Decomposed plot
- Correlogram plot
- Checking stationarity
## Data Analysis
```
# Load the data
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
data = pd.read_excel("C:/Users/admin/OneDrive/Documents/monthly_unemp.xlsx")

data.tail()

###########################################################################################
## Create a scatter plot
plt.scatter(data['Year'], data['Rates'])
plt.xlabel('Year')
plt.ylabel('Rates')
plt.title('Unemployment Rate by Year')

### Display the plot
plt.show()

############################################################################################
## Decompose the Data
import pandas as pd
from statsmodels.tsa.seasonal import seasonal_decompose
import matplotlib.pyplot as plt

### Load the data
data = data.set_index('Year')

### Perform seasonal decomposition
decomposition = seasonal_decompose(data['Rates'], model='additive')

### Extract the trend, seasonal, and residual components
trend = decomposition.trend
seasonal = decomposition.seasonal
residual = decomposition.resid

### Plot the decomposed components
plt.figure(figsize=(10, 8))

plt.subplot(411)
plt.plot(data['Rates'], label='Original')
plt.legend(loc='best')

plt.subplot(412)
plt.plot(trend, label='Trend')
plt.legend(loc='best')

plt.subplot(413)
plt.plot(seasonal, label='Seasonality')
plt.legend(loc='best')

plt.subplot(414)
plt.plot(residual, label='Residuals')
plt.legend(loc='best')

plt.tight_layout()
plt.show()

########################################################################################
## Correlogram Plot
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf


### Plot ACF and PACF
plt.figure(figsize=(10, 6))
plot_acf(data['Rates'], lags=30, color = 'black')
plot_pacf(data['Rates'], lags=30, color = 'black')

### Customize the plot
plt.xlabel('Lags')
plt.ylabel('Autocorrelation')
plt.title('Autocorrelation Function (ACF)')
plt.show()

########################################################################################
## Checking stationarity using ADF Test
import pandas as pd
from statsmodels.tsa.stattools import adfuller


### Perform the ADF test
result = adfuller(data['Rates'])

### Extract and print the test statistics
adf_statistic = result[0]
p_value = result[1]
critical_values = result[4]

print(f'ADF Statistic: {adf_statistic:.4f}')
print(f'p-value: {p_value:.4f}')
print('Critical Values:')
for key, value in critical_values.items():
    print(f'   {key}: {value:.4f}')

##########################################################################################
## Differencing
import pandas as pd
from statsmodels.tsa.stattools import adfuller
import matplotlib.pyplot as plt

### Perform differencing until data becomes stationary
is_stationary = False
differenced_data = data['Rates']

while not is_stationary:
    #### Perform differencing
    differenced_data = differenced_data.diff()
    differenced_data = differenced_data.dropna()
    
    #### Check stationarity
    result = adfuller(differenced_data)
    p_value = result[1]
    
    #### Set is_stationary flag based on p-value
    is_stationary = p_value < 0.05

### Print the differenced data
print(differenced_data)

### Remove the NaN values
differenced_data = differenced_data.dropna()

### Plot the differenced series
plt.figure(figsize=(10, 6))
plt.plot(differenced_data, color='black')  # Set the color to black
plt.xlabel('Year')
plt.xlabel('Year')
plt.ylabel('Differenced Rate')
plt.title('Differenced Series')

### Display the plot
plt.show()
#################################################################################################
## Check Stationarity for Differenced data
### Perform ADF test on the differenced series
result = adfuller(differenced_data)

### Extract and print the test statistics
adf_statistic = result[0]
p_value = result[1]
critical_values = result[4]

print(f'ADF Statistic: {adf_statistic:.4f}')
print(f'p-value: {p_value:.4f}')
print('Critical Values:')
for key, value in critical_values.items():
    print(f'   {key}: {value:.4f}')

####################################################################################################
## ARIMA Model

import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
ts_data = data['Rates']
### Fit ARIMA model
order = (0, 2, 1)
arima_model = ARIMA(ts_data, order=order)
arima_model_fit = arima_model.fit()

### Print model summary
print(arima_model_fit.summary())

### Forecast
forecast_steps = 36  # Adjust the forecast horizon as needed
arima_forecast = arima_model_fit.forecast(steps=forecast_steps)

### Print forecasted values
print("ARIMA Model - Forecasted Values:")
print(arima_forecast)

### Plot original data and forecasted values
plt.figure(figsize=(10, 6))
plt.plot(ts_data, label='Original Data')
plt.plot(arima_forecast, label='ARIMA Forecast')
plt.xlabel('Time')
plt.ylabel('Value')
plt.title('ARIMA(0,2,1) Forecast')
plt.legend()
plt.grid(True)
plt.show()

#############################################################################################################
## Residual Plots for ARIMA
import matplotlib.pyplot as plt
import seaborn as sns

### Get the residuals
residuals1 = arima_model_fit.resid

### Residual plots
plt.figure(figsize=(10, 6))
plt.subplot(2, 2, 1)
plt.plot(residuals1)
plt.xlabel('Time')
plt.ylabel('Residuals')
plt.title('Residuals vs. Time')

plt.subplot(2, 2, 2)
plt.scatter(ts_data, residuals1)
plt.xlabel('Observed Values')
plt.ylabel('Residuals')
plt.title('Residuals vs. Observed Values')

### Box plot of residuals
plt.subplot(2, 2, 3)
sns.boxplot(residuals1)
plt.xlabel('Residuals')
plt.title('Box Plot of Residuals')

plt.tight_layout()
plt.show()

#############################################################################################
## SARIMA MODEL
import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
import matplotlib.pyplot as plt


ts_data = data['Rates']

### Fit SARIMA model
order = (0, 2, 1)
seasonal_order = (2, 0, 2, 12)
sarima_model = SARIMAX(ts_data, order=order, seasonal_order=seasonal_order)
sarima_model_fit = sarima_model.fit()
### Print model summary
print(sarima_model_fit.summary())


### Forecast
forecast_steps = 36  # Adjust the forecast horizon as needed
sarima_forecast = sarima_model_fit.get_forecast(steps=forecast_steps)

### Extract forecasted values and confidence intervals
sarima_forecast_values = sarima_forecast.predicted_mean
confidence_intervals = sarima_forecast.conf_int()

### Plot original data and forecasted values
plt.style.use('default')

plt.figure(figsize=(10, 6))
plt.plot(ts_data, label='Original Data')
plt.plot(sarima_forecast_values, label='SARIMA Forecast')
plt.fill_between(confidence_intervals.index, confidence_intervals.iloc[:, 0], confidence_intervals.iloc[:, 1], alpha=0.3)
plt.xlabel('Time')
plt.ylabel('Value')
plt.title('SARIMA Forecast')
plt.legend()
plt.grid(True)
plt.show()

##############################################################################################################################
## Residual Plot for SARIMA
import matplotlib.pyplot as plt
import seaborn as sns

### Get the residuals
residuals = sarima_model_fit.resid

### Residual plots
plt.figure(figsize=(10, 6))
plt.subplot(2, 2, 1)
plt.plot(residuals)
plt.xlabel('Time')
plt.ylabel('Residuals')
plt.title('Residuals vs. Time')

plt.subplot(2, 2, 2)
plt.scatter(ts_data, residuals)
plt.xlabel('Observed Values')
plt.ylabel('Residuals')
plt.title('Residuals vs. Observed Values')

### Box plot of residuals
plt.subplot(2, 2, 3)
sns.boxplot(residuals)
plt.xlabel('Residuals')
plt.title('Box Plot of Residuals')



plt.tight_layout()
plt.show()

############################################################################################################
## HOLT WINTERS MODEL
import pandas as pd
import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import matplotlib.pyplot as plt

ts_data = data['Rates']

### Fit Holt-Winters model (multiplicative)
holtwinters_model = ExponentialSmoothing(ts_data, trend='mul', seasonal='mul')
holtwinters_model_fit = holtwinters_model.fit()

### Forecast
forecast_steps = 12  # Adjust the forecast horizon as needed
holtwinters_forecast = holtwinters_model_fit.forecast(steps=forecast_steps)

### Plot original data and forecasted values
plt.figure(figsize=(10, 6))
plt.plot(ts_data, label='Original Data')
plt.plot(holtwinters_forecast, label='Holt-Winters Forecast')
plt.xlabel('Time')
plt.ylabel('Value')
plt.title('Holt-Winters Multiplicative Forecast')
plt.legend()
plt.grid(True)
plt.show()

####################################################################################################
## Calculating Long Term and Short Term Forecasts for the 3 Models
import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing

### Data
ts_data = data['Rates']

### Split data into train and test sets
train_data = ts_data[:-12]  # Use the first (n-12) data points for training
test_data = ts_data[-12:]  # Use the last 12 data points for testing

### Fit and forecast using ARIMA model
arima_order = (0, 2, 1)
arima_model = ARIMA(train_data, order=arima_order)
arima_model_fit = arima_model.fit()
arima_forecast_short = arima_model_fit.forecast(steps=3)  # Short-term forecast (3 months)
arima_forecast_long = arima_model_fit.forecast(steps=36)  # Long-term forecast (3 years)

### Fit and forecast using SARIMA model
sarima_order = (0, 2, 1)
sarima_seasonal_order = (2, 0, 2, 12)
sarima_model = SARIMAX(train_data, order=sarima_order, seasonal_order=sarima_seasonal_order)
sarima_model_fit = sarima_model.fit()
sarima_forecast_short = sarima_model_fit.get_forecast(steps=3).predicted_mean  # Short-term forecast (3 months)
sarima_forecast_long = sarima_model_fit.get_forecast(steps=36).predicted_mean  # Long-term forecast (3 years)

### Fit and forecast using Holt-Winters model (multiplicative)
holtwinters_model = ExponentialSmoothing(train_data, trend='mul', seasonal='mul')
holtwinters_model_fit = holtwinters_model.fit()
holtwinters_forecast_short = holtwinters_model_fit.forecast(steps=3)  # Short-term forecast (3 months)
holtwinters_forecast_long = holtwinters_model_fit.forecast(steps=36)  # Long-term forecast (3 years)

### Calculate RMSE for short-term forecasts
arima_rmse_short = np.sqrt(np.mean((test_data[:3] - arima_forecast_short)**2))
sarima_rmse_short = np.sqrt(np.mean((test_data[:3] - sarima_forecast_short)**2))
holtwinters_rmse_short = np.sqrt(np.mean((test_data[:3] - holtwinters_forecast_short)**2))

### Calculate MAPE for short-term forecasts
arima_mape_short = np.mean(np.abs((test_data[:3] - arima_forecast_short) / test_data[:3])) * 100
sarima_mape_short = np.mean(np.abs((test_data[:3] - sarima_forecast_short) / test_data[:3])) * 100
holtwinters_mape_short = np.mean(np.abs((test_data[:3] - holtwinters_forecast_short) / test_data[:3])) * 100

### Calculate RMSE for long-term forecasts
arima_rmse_long = np.sqrt(np.mean((test_data - arima_forecast_long)**2))
sarima_rmse_long = np.sqrt(np.mean((test_data - sarima_forecast_long)**2))
holtwinters_rmse_long = np.sqrt(np.mean((test_data - holtwinters_forecast_long)**2))

### Calculate MAPE for long-term forecasts
arima_mape_long = np.mean(np.abs((test_data - arima_forecast_long) / test_data)) * 100
sarima_mape_long = np.mean(np.abs((test_data - sarima_forecast_long) / test_data)) * 100
holtwinters_mape_long = np.mean(np.abs((test_data - holtwinters_forecast_long) / test_data)) * 100

### Print RMSE and MAPE values for short-term forecasts
print("Short-term Forecasts (3 months):")
print("ARIMA Model - RMSE: {:.2f}, MAPE: {:.2f}%".format(arima_rmse_short, arima_mape_short))
print("SARIMA Model - RMSE: {:.2f}, MAPE: {:.2f}%".format(sarima_rmse_short, sarima_mape_short))
print("Holt-Winters Model - RMSE: {:.2f}, MAPE: {:.2f}%".format(holtwinters_rmse_short, holtwinters_mape_short))
print()

### Print RMSE and MAPE values for long-term forecasts
print("Long-term Forecasts (3 years):")
print("ARIMA Model - RMSE: {:.2f}, MAPE: {:.2f}%".format(arima_rmse_long, arima_mape_long))
print("SARIMA Model - RMSE: {:.2f}, MAPE: {:.2f}%".format(sarima_rmse_long, sarima_mape_long))
print("Holt-Winters Model - RMSE: {:.2f}, MAPE: {:.2f}%".format(holtwinters_rmse_long, holtwinters_mape_long))

#########################################################################################################################
## Ploting Short Term Forecasts

### Combine the forecasts into a DataFrame
forecasts = pd.DataFrame({
    'ARIMA': arima_forecast_short,
    'SARIMA': sarima_forecast_short,
    'Holt-Winters': holtwinters_forecast_short
})

### Plot the forecasts
plt.figure(figsize=(10, 6))
plt.plot(ts_data, label='Original')
for col in forecasts.columns:
    plt.plot(forecasts[col], label=col)
plt.title('ARIMA, SARIMA, and Holt-Winters Forecasts')
plt.xlabel('Time')
plt.ylabel('Value')
plt.legend()
plt.show()

#####################################################################################################
## Ploting the Long Term Forecasts

# Combine the forecasts into a DataFrame
forecasts = pd.DataFrame({
    'ARIMA': arima_forecast_long,
    'SARIMA': sarima_forecast_long,
    'Holt-Winters': holtwinters_forecast_long
})

# Plot the forecasts
plt.figure(figsize=(10, 6))
plt.plot(ts_data, label='Original')
for col in forecasts.columns:
    plt.plot(forecasts[col], label=col)
plt.title('ARIMA, SARIMA, and Holt-Winters Forecasts')
plt.xlabel('Time')
plt.ylabel('Value')
plt.legend()
plt.show()

#######################################################################################################
```
#
