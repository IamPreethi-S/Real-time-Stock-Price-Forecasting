import streamlit as st
import pandas as pd
import yfinance as yf
from models.arima_model import train_arima, predict_arima
from models.fbprophet_model import train_fbprophet, predict_fbprophet
from models.lstm_model import train_lstm, predict_lstm
from models.arima_lstm_hybrid_model import train_arima_lstm, predict_arima_lstm
from utils.evaluation_metrics import calculate_rmse, calculate_mse, calculate_mape, plot_predictions
import matplotlib.pyplot as plt

def fetch_data(ticker):
    data = yf.download(ticker, start="2015-01-01", end="2023-01-01")
    data.reset_index(inplace=True)
    return data

def align_predictions(y_true, y_pred):
    min_len = min(len(y_true), len(y_pred))
    return y_true[:min_len], y_pred[:min_len]

# Streamlit app
st.title('Stock Price Prediction')
st.sidebar.title('Settings')

# Sidebar options
ticker = st.sidebar.text_input('Stock Ticker', value='AAPL')
period = st.sidebar.selectbox('Prediction Period', ['1 month', '6 months', '1 year', '5 years'])

# Mapping periods to steps
periods_map = {
    '1 month': 30,
    '6 months': 182,
    '1 year': 365,
    '5 years': 1825
}
steps = periods_map[period]

if st.sidebar.button('Predict'):
    with st.spinner('Fetching data and predicting...'):
        # Fetch data
        data = fetch_data(ticker)
        
        # Train models
        st.write("Training ARIMA model...")
        arima_model = train_arima(data)
        
        st.write("Training FbProphet model...")
        fbprophet_model = train_fbprophet(data)
        
        st.write("Training LSTM model...")
        lstm_model, lstm_scaler = train_lstm(data)
        
        st.write("Training ARIMA-LSTM hybrid model...")
        arima_model, hybrid_lstm_model, hybrid_scaler = train_arima_lstm(data)
        
        # Predictions
        st.write(f"\nPredicting for {period}...")
        
        arima_predictions = predict_arima(arima_model, steps)
        fbprophet_predictions = predict_fbprophet(fbprophet_model, steps)['yhat'].values
        lstm_predictions = predict_lstm(lstm_model, lstm_scaler, data, steps)[-steps:].flatten()
        hybrid_predictions = predict_arima_lstm(arima_model, hybrid_lstm_model, hybrid_scaler, data, steps)[-steps:].flatten()
        
        # Evaluation
        if steps <= len(data):
            y_true = data['Close'].values[-steps:]
            
            y_true_arima, arima_predictions = align_predictions(y_true, arima_predictions)
            y_true_fbprophet, fbprophet_predictions = align_predictions(y_true, fbprophet_predictions)
            y_true_lstm, lstm_predictions = align_predictions(y_true, lstm_predictions)
            y_true_hybrid, hybrid_predictions = align_predictions(y_true, hybrid_predictions)
            
            st.write("\nEvaluation Results:")
            st.write(f"ARIMA Model - RMSE: {calculate_rmse(y_true_arima, arima_predictions)}, MSE: {calculate_mse(y_true_arima, arima_predictions)}, MAPE: {calculate_mape(y_true_arima, arima_predictions)}")
            st.write(f"FbProphet Model - RMSE: {calculate_rmse(y_true_fbprophet, fbprophet_predictions)}, MSE: {calculate_mse(y_true_fbprophet, fbprophet_predictions)}, MAPE: {calculate_mape(y_true_fbprophet, fbprophet_predictions)}")
            st.write(f"LSTM Model - RMSE: {calculate_rmse(y_true_lstm, lstm_predictions)}, MSE: {calculate_mse(y_true_lstm, lstm_predictions)}, MAPE: {calculate_mape(y_true_lstm, lstm_predictions)}")
            st.write(f"ARIMA-LSTM Hybrid Model - RMSE: {calculate_rmse(y_true_hybrid, hybrid_predictions)}, MSE: {calculate_mse(y_true_hybrid, hybrid_predictions)}, MAPE: {calculate_mape(y_true_hybrid, hybrid_predictions)}")
        
            # Plot Predictions
            fig, ax = plt.subplots(figsize=(14, 7))
            ax.plot(range(len(y_true)), y_true, label='True Prices', color='black', linestyle='-', marker='o')
            ax.plot(range(len(arima_predictions)), arima_predictions, label='ARIMA Predictions', color='blue', linestyle='--', marker='x')
            ax.plot(range(len(fbprophet_predictions)), fbprophet_predictions, label='FbProphet Predictions', color='green', linestyle='-.', marker='v')
            ax.plot(range(len(lstm_predictions)), lstm_predictions, label='LSTM Predictions', color='red', linestyle=':', marker='s')
            ax.plot(range(len(hybrid_predictions)), hybrid_predictions, label='ARIMA-LSTM Predictions', color='purple', linestyle='-', marker='d')
            ax.set_title(f'Stock Price Predictions over {period}')
            ax.set_xlabel('Days')
            ax.set_ylabel('Prices')
            ax.legend()
            ax.grid(True)
            st.pyplot(fig)
        else:
            st.write("Future period, cannot evaluate.")
            
            # Plot future predictions without evaluation
            fig, ax = plt.subplots(figsize=(14, 7))
            ax.plot(range(len(data['Close'].values)), data['Close'].values, label='True Prices', color='black', linestyle='-', marker='o')
            ax.plot(range(len(arima_predictions)), arima_predictions, label='ARIMA Predictions', color='blue', linestyle='--', marker='x')
            ax.plot(range(len(fbprophet_predictions)), fbprophet_predictions, label='FbProphet Predictions', color='green', linestyle='-.', marker='v')
            ax.plot(range(len(lstm_predictions)), lstm_predictions, label='LSTM Predictions', color='red', linestyle=':', marker='s')
            ax.plot(range(len(hybrid_predictions)), hybrid_predictions, label='ARIMA-LSTM Predictions', color='purple', linestyle='-', marker='d')
            ax.set_title(f'Stock Price Predictions over {period}')
            ax.set_xlabel('Days')
            ax.set_ylabel('Prices')
            ax.legend()
            ax.grid(True)
            st.pyplot(fig)
