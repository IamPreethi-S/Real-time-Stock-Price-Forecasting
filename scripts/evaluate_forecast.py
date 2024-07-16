import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Load the original dataset
df = pd.read_csv('../data/prepared_stock_data.csv')
df['ds'] = pd.to_datetime(df['ds'])

# Load the forecast data
forecast = pd.read_csv('../data/forecast_data.csv')
forecast['ds'] = pd.to_datetime(forecast['ds'])

# Split the data into training and test sets
train_size = int(len(df) * 0.8)
train, test = df.iloc[:train_size], df.iloc[train_size:]

# Extract the test period from the forecast
test_forecast = forecast[forecast['ds'].isin(test['ds'])]

# Evaluate predictions
predictions = test_forecast['yhat'].values
true_values = test['y'].values

rmse = mean_squared_error(true_values, predictions, squared=False)
mae = mean_absolute_error(true_values, predictions)
mape = (abs(true_values - predictions) / true_values).mean() * 100
r2 = r2_score(true_values, predictions)

print(f'RMSE: {rmse}')
print(f'MAE: {mae}')
print(f'MAPE: {mape}')
print(f'R²: {r2}')
