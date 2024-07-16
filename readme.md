# Real-time Stock Price Prediction

This project provides a real-time stock price prediction dashboard using Streamlit and the Prophet forecasting model. It fetches stock data from Yahoo Finance and generates future stock price forecasts.


## Steps to Run

1. **Set up the virtual environment**:

2. **Install dependencies**:
    ```
    pip install -r requirements.txt
    ```
3. **Generate forecasts**:
    python generate_forecasts.py

4. **Evaluate forecasts**:

    python evaluate_forecasts.py

5. **Run the model training script**:
    streamlit run app.py


## Features

- **Real-time Stock Price Prediction**: Input stock tickers and date ranges to fetch historical data and generate forecasts.
- **Interactive Dashboard**: Visualize historical data, forecast data, and model components using Streamlit.
- **Data Fetching**: Integrates with Yahoo Finance to fetch up-to-date stock data.
- **Model Training and Evaluation**: Uses Prophet for time series forecasting and provides metrics such as RMSE, MAE, and MAPE.
