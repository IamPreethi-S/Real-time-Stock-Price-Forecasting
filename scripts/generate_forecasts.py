import pandas as pd
import matplotlib.pyplot as plt
import joblib
from prophet import Prophet

# Load the trained model
model = joblib.load('../models/prophet_model.joblib')

# Create a DataFrame for future predictions (next 365 days)
future = model.make_future_dataframe(periods=365)

# Generate forecasts
forecast = model.predict(future)

# Save the forecast data to a CSV file
forecast.to_csv('../data/forecast_data.csv', index=False)

# Visualize the forecasts
fig1 = model.plot(forecast)
plt.title('Stock Price Forecast')
plt.xlabel('Date')
plt.ylabel('Stock Price')
plt.grid(True)
plt.show()

# Plot forecast components
fig2 = model.plot_components(forecast)
plt.show()
