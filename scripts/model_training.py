import pandas as pd
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics
from prophet.plot import plot_cross_validation_metric
import joblib
import json
from prophet.serialize import model_to_json, model_from_json

# Load the prepared data
df = pd.read_csv('../data/prophet_ready_data.csv')

# Initialize the Prophet model
model = Prophet()

# Train the model
model.fit(df)

# Perform cross-validation with fewer processes
df_cv = cross_validation(
    model, 
    initial='730 days', 
    period='180 days', 
    horizon='30 days', 
    parallel="threads"  # Use "threads" instead of "processes"
)

# Calculate performance metrics
df_p = performance_metrics(df_cv)
print(df_p[['horizon', 'mae', 'rmse', 'mape']].head())

# Plot cross-validation results
fig = plot_cross_validation_metric(df_cv, metric='mae')
fig.suptitle('MAE by Forecast Horizon')
fig.show()

# Save the trained model
joblib.dump(model, '../models/prophet_model.joblib')
print("Model saved to ../models/prophet_model.joblib")

# Serialize the model for future use
with open('../models/serialized_model.json', 'w') as fout:
    json.dump(model_to_json(model), fout)
print("Serialized model saved to ../models/serialized_model.json")
