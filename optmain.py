# main.py
import pandas as pd
from models.arima_model import train_arima, predict_arima
from models.fbprophet_model import train_fbprophet, predict_fbprophet
from models.lstm_model import train_lstm, predict_lstm
from models.arima_lstm_hybrid_model import train_arima_lstm, predict_arima_lstm
from utils.evaluation_metrics import calculate_rmse, calculate_mse, calculate_mape, plot_predictions

def align_predictions(y_true, y_pred):
    min_len = min(len(y_true), len(y_pred))
    return y_true[:min_len], y_pred[:min_len]

if __name__ == "__main__":
    data = pd.read_csv("data/stock_data.csv")
    
    # Train Models
    print("Training ARIMA model...")
    arima_model = train_arima(data)
    
    print("Training FbProphet model...")
    fbprophet_model = train_fbprophet(data)
    
    print("Training LSTM model...")
    lstm_model, lstm_scaler = train_lstm(data)
    
    print("Training ARIMA-LSTM hybrid model...")
    arima_model, hybrid_lstm_model, hybrid_scaler = train_arima_lstm(data)
    
    # Prediction periods
    prediction_periods = {
        "1_month": 30,
        "6_months": 182,
        "1_year": 365,
        "5_years": 1825
    }
    
    for period_name, steps in prediction_periods.items():
        print(f"\nPredicting for {period_name}...")
        
        print("Predicting with ARIMA model...\n")
        arima_predictions = predict_arima(arima_model, steps)
        
        print("Predicting with FbProphet model...\n")
        fbprophet_predictions = predict_fbprophet(fbprophet_model, steps)['yhat'].values
        
        print("Predicting with LSTM model...\n")
        lstm_predictions = predict_lstm(lstm_model, lstm_scaler, data, steps)[-steps:].flatten()
        
        print("Predicting with ARIMA-LSTM hybrid model...\n")
        hybrid_predictions = predict_arima_lstm(arima_model, hybrid_lstm_model, hybrid_scaler, data, steps)[-steps:].flatten()
        
        if steps <= len(data):
            y_true = data['Close'].values[-steps:]
            
            y_true_arima, arima_predictions = align_predictions(y_true, arima_predictions)
            y_true_fbprophet, fbprophet_predictions = align_predictions(y_true, fbprophet_predictions)
            y_true_lstm, lstm_predictions = align_predictions(y_true, lstm_predictions)
            y_true_hybrid, hybrid_predictions = align_predictions(y_true, hybrid_predictions)
            
            print("\nEvaluation Results:")
            print(f"ARIMA Model - RMSE: {calculate_rmse(y_true_arima, arima_predictions)}, MSE: {calculate_mse(y_true_arima, arima_predictions)}, MAPE: {calculate_mape(y_true_arima, arima_predictions)}\n")
            print(f"FbProphet Model - RMSE: {calculate_rmse(y_true_fbprophet, fbprophet_predictions)}, MSE: {calculate_mse(y_true_fbprophet, fbprophet_predictions)}, MAPE: {calculate_mape(y_true_fbprophet, fbprophet_predictions)}\n")
            print(f"LSTM Model - RMSE: {calculate_rmse(y_true_lstm, lstm_predictions)}, MSE: {calculate_mse(y_true_lstm, lstm_predictions)}, MAPE: {calculate_mape(y_true_lstm, lstm_predictions)}\n")
            print(f"ARIMA-LSTM Hybrid Model - RMSE: {calculate_rmse(y_true_hybrid, hybrid_predictions)}, MSE: {calculate_mse(y_true_hybrid, hybrid_predictions)}, MAPE: {calculate_mape(y_true_hybrid, hybrid_predictions)}\n")
        
            # Plot Predictions
            plot_predictions(y_true, arima_predictions, fbprophet_predictions, lstm_predictions, hybrid_predictions, steps)
        else:
            print("Future period, cannot evaluate.")
            
            # Plot future predictions without evaluation
            plot_predictions(data['Close'].values, arima_predictions, fbprophet_predictions, lstm_predictions, hybrid_predictions, steps)
