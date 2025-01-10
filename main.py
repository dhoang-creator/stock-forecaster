import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns 
from statsmodels.tsa.arima.model import ARIMA 
from statsmodels.tsa.stattools import adfuller 
import yfinance as yf 

# Dataset loading
ticker = 'AAPL'
data = yf.download(ticker, start='2010-01-01', end='2023-01-01')

# You need the 'Close' price for forecasting
stock_data = data['Close']

# First, you should visualise the data
plt.figure(figsize=(10,6))
plt.plot(stock_data)
plt.title(f'{ticker} Stock Prices')
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.show()

# Stationarity Test
# Remember that if the mean and variance are constant over time, then the data is stationarity
def adf_test(series):
    result = adfuller(series)
    print('ADF Statistic:', result[0])
    print('p-value', result[1])
    print('Critical Values:')
    for key, value in result[4].items():
        print(f' {key}: {value}')

adf_test(stock_data)

# Differencing the data
stock_data_diff = stock_data.diff().dropna()

adf_test(stock_data_diff)

# ARIMA Modelling

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# Plot ACF and PACF
fig, ax = plt.subplots(1, 2, figsize=(16,6))
plot_acf(stock_data_diff, lags=40, ax=ax[0])
plot_pacf(stock_data_diff, lags=40, ax=ax[1])
plt.show


model = ARIMA(stock_data, order=(5, 1, 0)) # ARIMA(p, d, q)
model_fit = model.fit()

print(model_fit.summary())


# Then you need to make predictions and forecast over a certain time period i.e. over the next 10 days
forecast_steps = 10
forecast = model_fit.forecast(steps=forecast_steps)
forecast_index = pd.date_range(stock_data.index[-1], periods=forecast_steps+1, freq='B')[1:]

# Plot the forecasted values
plt.figure(figsize=(10,6))
plt.plot(stock_data.index, stock_data, label='Historical Data')
plt.plot(forecast_index, forecast, label='Forecasted Data', colors='red')
plt.title(f'{ticker} Stock Price Forecast')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.show()
