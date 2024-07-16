# app/app.py

from flask import Flask, request, jsonify
from prophet import Prophet
from prophet.serialize import model_from_json
import pandas as pd
import json
import os

app = Flask(__name__)

# Load the trained model
model_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'serialized_model.json')
with open(model_path, 'r') as fin:
    model = model_from_json(json.load(fin))

@app.errorhandler(400)
def bad_request(error):
    return jsonify({"error": "Bad request", "message": str(error)}), 400

@app.errorhandler(500)
def internal_error(error):
    return jsonify({"error": "Internal server error", "message": str(error)}), 500

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get JSON data from request
        data = request.get_json()

        # Validate input
        if 'periods' not in data:
            return jsonify({"error": "Missing 'periods' in request data"}), 400

        periods = int(data['periods'])
        if periods <= 0:
            return jsonify({"error": "'periods' must be a positive integer"}), 400

        # Make future dataframe for prediction
        future = model.make_future_dataframe(periods=periods, freq='D')
        
        # Make prediction
        forecast = model.predict(future)

        # Prepare the response
        response = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(periods).to_dict('records')
        
        return jsonify(response)

    except Exception as e:
        app.logger.error(f"An error occurred: {str(e)}")
        return jsonify({"error": "An unexpected error occurred"}), 500

if __name__ == '__main__':
    app.run(debug=True)