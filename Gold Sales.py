import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt

from sklearn.linear_model import LinearRegression
from statsmodels.tsa.api import ExponentialSmoothing, SimpleExpSmoothing, Holt

import warnings
warnings.filterwarnings('ignore')

# Load the data
df = pd.read_csv(r'Downloads/Data.csv')
print(df.head())
print(df.shape)

# Display data range
print(f"Data range available from - {df['Date'].iloc[0]} to {df['Date'].iloc[-1]}")

# Create a date range for the 'Month' column (assuming monthly frequency)
date = pd.date_range(start='1/1/1950', end='8/1/2020', freq='M')

# Add the 'Month' column to the dataframe and drop the old 'Date' column
df['Month'] = date
df.drop('Date', axis=1, inplace=True)

# Set 'Month' as the index
df = df.set_index('Month')
print(df.head())

# Plot the time series of gold prices
df.plot(figsize=(20, 8))
plt.title("Gold Prices Since 1950 and Onwards")
plt.xlabel('Year')
plt.ylabel('Price')
plt.grid()
plt.show()

# Summary statistics of the dataset
print(round(df.describe(), 3))

# Boxplot to show the distribution of gold prices by year
_, ax = plt.subplots(figsize=(20, 8))
sns.boxplot(x=df.index.year, y=df.values[:, 0], ax=ax)
plt.title("Gold Prices Since 1950 and Onwards")
plt.xlabel('Year')
plt.ylabel('Price')
plt.xticks(rotation=90)
plt.grid()
plt.show()

# Month-wise plot of gold prices
from statsmodels.graphics.tsaplots import month_plot

fig, ax = plt.subplots(figsize=(20, 8))
month_plot(df, ylabel='Gold Price', ax=ax)
plt.title("Gold Price Monthly Since 1950 and Onwards")
plt.xlabel('Month')
plt.ylabel('Price')
plt.grid()
plt.show()

# Boxplot by month
_, ax = plt.subplots(figsize=(22, 8))
sns.boxplot(x=df.index.month_name(), y=df.values[:, 0], ax=ax)
plt.title("Gold Price Monthly Since 1950 and Onwards")
plt.xlabel('Month')
plt.ylabel('Price')
plt.grid()
plt.show()

# Resample data and plot annual, quarterly, and decade-based averages
df_yearly_avg = df.resample('A').mean()
df_yearly_avg.plot(figsize=(12, 6))
plt.title("Average Gold Price Yearly Since 1950 and Onwards")
plt.xlabel('Year')
plt.ylabel('Average Price')
plt.grid()
plt.show()

df_quarterly_avg = df.resample('Q').mean()
df_quarterly_avg.plot(figsize=(12, 6))
plt.title("Average Gold Price Quarterly Since 1950 and Onwards")
plt.xlabel('Quarter')
plt.ylabel('Average Price')
plt.grid()
plt.show()

df_decade_avg = df.resample('10Y').mean()
df_decade_avg.plot(figsize=(12, 6))
plt.title("Average Gold Price Decade Since 1950 and Onwards")
plt.xlabel('Decade')
plt.ylabel('Average Price')
plt.grid()
plt.show()

# Calculate the coefficient of variation (CV) and plot it
df_1 = df.groupby(df.index.year).mean().rename(columns={'Price': 'Mean'})
df_1 = df_1.merge(df.groupby(df.index.year).std().rename(columns={'Price': 'Std'}), left_index=True, right_index=True)
df_1['Cov'] = ((df_1['Std'] / df_1['Mean']) * 100).round(2)

fig, ax = plt.subplots(figsize=(15, 10))
df_1['Cov'].plot(ax=ax)
plt.title("Coefficient of Variation (CV) of Gold Prices Yearly Since 1950")
plt.xlabel('Year')
plt.ylabel('CV in %')
plt.grid()
plt.show()

# Split the data into training and test sets based on year
train = df[df.index.year <= 2015]
test = df[df.index.year > 2015]

# Plot the training and test data
train['Price'].plot(figsize=(13, 5), fontsize=15)
test['Price'].plot(figsize=(13, 5), fontsize=15)
plt.grid()
plt.legend(['Training Data', 'Test Data'])
plt.show()

# Create time variables for linear regression
train_time = np.arange(len(train)) + 1
test_time = np.arange(len(test)) + len(train) + 1

# Create copies of the training and test sets for linear regression
LR_train = train.copy()
LR_test = test.copy()

# Add the 'time' column to both the training and test sets
LR_train['time'] = train_time
LR_test['time'] = test_time

# Fit a linear regression model using training data
lr = LinearRegression()
lr.fit(LR_train['time'].values.reshape(-1, 1), LR_train['Price'].values)

# Predict prices on the test data
predictions = lr.predict(LR_test['time'].values.reshape(-1, 1))

# Plot actual vs predicted prices
plt.figure(figsize=(13, 5))
plt.plot(train.index, train['Price'], label='Training Data')
plt.plot(test.index, test['Price'], label='Actual Test Data')
plt.plot(test.index, predictions, label='Predicted Test Data', linestyle='--')
plt.title("Gold Price Prediction Using Linear Regression")
plt.xlabel('Year')
plt.ylabel('Price')
plt.legend()
plt.grid()
plt.show()

# Time Series Forecasting with Exponential Smoothing (Holt-Winters)
model = ExponentialSmoothing(train['Price'], seasonal='mul', seasonal_periods=12).fit()
forecast = model.forecast(steps=len(test))

# Plot actual vs forecasted prices
plt.figure(figsize=(13, 5))
plt.plot(train.index, train['Price'], label='Training Data')
plt.plot(test.index, test['Price'], label='Actual Test Data')
plt.plot(test.index, forecast, label='Forecasted Test Data', linestyle='--')
plt.title("Gold Price Forecast Using Holt-Winters Method")
plt.xlabel('Year')
plt.ylabel('Price')
plt.legend()
plt.grid()
plt.show()

# Calculate Mean Squared Error for the forecast
from sklearn.metrics import mean_squared_error
mse = mean_squared_error(test['Price'], forecast)
print(f"Mean Squared Error (MSE) for Holt-Winters Forecast: {mse:.2f}")