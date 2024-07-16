import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the prepared data
df = pd.read_csv('../data/prepared_stock_data.csv')

# Basic Statistical Analysis
print("Basic Statistical Analysis:")
print(df.describe())

# Plot the time series
plt.figure(figsize=(15, 7))
plt.plot(df['ds'], df['y'])
plt.title('Stock Prices Over Time')
plt.xlabel('Date')
plt.ylabel('Stock Price')
plt.grid(True)
plt.show()

# Check for seasonality
df['ds'] = pd.to_datetime(df['ds'])
df['year'] = df['ds'].dt.year
df['month'] = df['ds'].dt.month
df['day'] = df['ds'].dt.day
df['day_of_week'] = df['ds'].dt.dayofweek
df['hour'] = df['ds'].dt.hour

# Plot average stock price by year
plt.figure(figsize=(12, 6))
sns.boxplot(x='year', y='y', data=df)
plt.title('Boxplot of Stock Prices by Year')
plt.show()

# Plot average stock price by month
plt.figure(figsize=(12, 6))
sns.boxplot(x='month', y='y', data=df)
plt.title('Boxplot of Stock Prices by Month')
plt.show()

# Plot average stock price by day of week
plt.figure(figsize=(12, 6))
sns.boxplot(x='day_of_week', y='y', data=df)
plt.title('Boxplot of Stock Prices by Day of Week')
plt.show()

# Plot average stock price by hour
plt.figure(figsize=(12, 6))
sns.boxplot(x='hour', y='y', data=df)
plt.title('Boxplot of Stock Prices by Hour')
plt.show()

# Save the enhanced dataframe with additional columns
df.to_csv('../data/enhanced_stock_data.csv', index=False)
print("Enhanced data saved to data/enhanced_stock_data.csv")
