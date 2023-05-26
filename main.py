import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import sklearn
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures



# TODO Define Variables

# Define Period
period = "S"
# Autocrrelation threshold
auto_corr_lim = 0.5
# Autocorrelation Lag
auto_corr_lag = 1
# Set the window size for the rolling autocorrelation
window_size = 15
# Moving Average window
window = 1


# TODO Carry on:
# Load the data from the excel file
if period == "Q":
    df_y = pd.read_excel('Conversion.xlsx', sheet_name='y variables - Q', index_col=0)
    df_x = pd.read_excel('Conversion.xlsx', sheet_name='x variables - Q', index_col=0)
elif period == "S":
    df_y = pd.read_excel('Conversion.xlsx', sheet_name='y variables - S', index_col=0)
    df_x = pd.read_excel('Conversion.xlsx', sheet_name='x variables - S', index_col=0)
else:
    df_y = pd.read_excel('Conversion.xlsx', sheet_name='y variables', index_col=0)
    df_x = pd.read_excel('Conversion.xlsx', sheet_name='x variables', index_col=0)

# Define the start and finish dates
start = pd.to_datetime('1999-12-31')
finish = pd.to_datetime('2023-03-31')

# Filter the data to only include the specified time period
df_y_returns = df_y.loc[start:finish].pct_change().dropna()
df_x_returns = df_x.loc[start:finish].pct_change().dropna()

# Smooth public market returns
df_x_returns = df_x_returns.rolling(window).mean()

for col in df_x_returns.columns:
    first_non_nan = df_x_returns[col].dropna().iloc[0]
    df_x_returns[col].fillna(value=first_non_nan, inplace=True)



# Create an empty dataframe to store the results
df_autocorr = pd.DataFrame(columns=df_y_returns.columns).dropna()


# compute the rolling autocorrelation for each column
for col in df_y_returns.columns:
    autocorr = df_y_returns[col].rolling(window_size).apply(lambda x: x.autocorr(lag=auto_corr_lag))
    df_autocorr[col] = autocorr


# Replace NaN values in autocorr_df with the first non-NaN value in each column
for col in df_autocorr.columns:
    first_non_nan = df_autocorr[col].dropna().iloc[0]
    df_autocorr[col].fillna(value=first_non_nan, inplace=True)

# Print the last aucorrelation values
last_autocorr = df_autocorr.iloc[-1]
print(f'Past five years of autocorrelation: \n {last_autocorr}')


df_autocorr = df_autocorr.where((df_autocorr < -auto_corr_lim) | (df_autocorr > auto_corr_lim), other=0)


# Create a new DataFrame to store the results
result_df = pd.DataFrame(index=df_y_returns.index, columns=df_y_returns.columns)


# Unsmoothing returns if autocorrelated
for i in range(1, len(df_y_returns)):
    result = (df_y_returns.iloc[i] - df_y_returns.iloc[i-1] * df_autocorr.iloc[i]) / (1 - df_autocorr.iloc[i])
    result_df.iloc[i] = result


# Calculate the cumulative return, average return, and volatility for every column
cumulative_returns_y = (1 + df_y_returns).cumprod() - 1
cumulative_returns_x = (1 + df_x_returns).cumprod() - 1
cumulative_returns = pd.concat([cumulative_returns_x, cumulative_returns_y], axis=1)

average_returns = df_y_returns.mean() * 12
volatility = df_y_returns.std() * (12**0.5)

# Output the results
print('Average Returns:')
print(average_returns)
print('Volatility:')
print(volatility)
print("'Sharpe' Ratio:")
print(average_returns/volatility)
print(df_x_returns.std() * (12**0.5))


# Create an iterative loop where every column from the returns table is linear regressed with the "dependent variables"
# table, assuming that the intercept is equal to "0"

for col in df_y_returns.columns:
    model = sm.OLS(df_y_returns[col], df_x_returns, hasconst=False)
    results = model.fit()
    print(f'Returns for {col} regressed against dependent variables:\n{results.summary()}\n')


# # Define the non-linear function to be fitted
# def non_linear_func(X, a, b, c):
#     return a * X[:, 0]**2 + b * X[:, 1]**2 + c * X[:, 2]
#
# # Set the constant to zero
# fit_intercept = False
#
# # Create polynomial features up to degree 2
# poly = PolynomialFeatures(degree=2, include_bias=False)
# X_poly = poly.fit_transform(df_x_returns)
#
# # Fit the non-linear regression model
# model = LinearRegression(fit_intercept=fit_intercept)
# model.fit(X_poly, df_y_returns)
#
# # Extract the coefficients of the non-linear function
# a, b, c = model.coef_






# Output graphs
# plt.scatter(df_y_returns[col], df_x_returns[:, 0])
# plt.show()
cumulative_returns.plot(title='Cumulative Returns')

# Provide the correlation table of all the columns
correlation_table = df_y_returns.corr()
