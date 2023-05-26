import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import openpyxl

# Load the data from the excel file
df = pd.read_excel('London.xlsx', sheet_name='HPI Comparison', index_col=0)

# Set the start and finish dates
start = pd.to_datetime('2012-01-01')
finish = pd.to_datetime('2022-12-01')

# Filter the DataFrame to only include data between the start and finish dates
df = df.loc[start:finish]

# Calculate the returns
returns = df.pct_change().dropna()

# Calculate the cumulative returns
cumulative_returns = (1 + returns).cumprod() - 1

# Calculate the average returns
avg_returns = returns.mean()*12

# Calculate the volatility
volatility = returns.std() * (12**0.5)

# Output the results
print('Average Returns:')
print(avg_returns)
print('Volatility:')
print(volatility)


# Plot the cumulative return paths
cumulative_returns.plot(title='Cumulative Returns')

# Calculate the correlation table
corr_table = returns.corr()

# Output the correlation table
print('Correlation Table:')
print(corr_table)


