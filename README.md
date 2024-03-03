# Modelling-Unemployment-Rates-using-Time-Series-Models
### Project Overview
Using Time Series models such as ARIMA, SARIMA, and HOLT WINTERS models to model and forecast unemployment rates.
### Data Source
- Kenya's monthly unemployment data 
### Tools
- Rstudio

### Data Cleaning
The data is clean for analysis

### Exploratory Data Analysis
- Descriptive statistics
- Trend chart
- Decomposed plot
- Correlogram plot
- Checking stationarity
### Data Analysis
```
# Load the data
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
data = pd.read_excel("C:/Users/admin/OneDrive/Documents/monthly_unemp.xlsx")

data.tail()

###########################################################################################
# Create a scatter plot
plt.scatter(data['Year'], data['Rates'])
plt.xlabel('Year')
plt.ylabel('Rates')
plt.title('Unemployment Rate by Year')

# Display the plot
plt.show()

############################################################################################
# Decompose the Data
import pandas as pd
from statsmodels.tsa.seasonal import seasonal_decompose
import matplotlib.pyplot as plt

# Load the data
data = data.set_index('Year')

# Perform seasonal decomposition
decomposition = seasonal_decompose(data['Rates'], model='additive')

# Extract the trend, seasonal, and residual components
trend = decomposition.trend
seasonal = decomposition.seasonal
residual = decomposition.resid

# Plot the decomposed components
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
