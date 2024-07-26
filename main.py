import pandas as pd
from models.arima_model import train_arima, predict_arima
from models.fbprophet_model import train_fbprophet, predict_fbprophet
from models.lstm_model import train_lstm, predict_lstm
from models.arima_lstm_hybrid_model import train_arima_lstm, predict_arima_lstm
from utils.evaluation_metrics import calculate_rmse, calculate_mse, calculate_mape, plot_predictions

if __name__ == "__main__":
    # Load the data
    data = pd.read_csv("data/stock_data.csv")
    
    # Train Models
    print("Training ARIMA model...\n")
    arima_model = train_arima(data)
    
    print("Training FbProphet model...")
    fbprophet_model = train_fbprophet(data)
    
    print("Training LSTM model...")
    lstm_model, lstm_scaler = train_lstm(data)
    
    print("Training ARIMA-LSTM hybrid model...")
    arima_model, hybrid_lstm_model, hybrid_scaler = train_arima_lstm(data)
    
    # Predictions
    steps = 10  # Number of days to predict
    print("Predicting with ARIMA model...")
    arima_predictions = predict_arima(arima_model, steps)
    
    print("Predicting with FbProphet model...")
    fbprophet_predictions = predict_fbprophet(fbprophet_model, steps)['yhat'].values
    
    print("Predicting with LSTM model...")
    lstm_predictions = predict_lstm(lstm_model, lstm_scaler, data, steps)[-steps:]
    
    print("Predicting with ARIMA-LSTM hybrid model...")
    hybrid_predictions = predict_arima_lstm(arima_model, hybrid_lstm_model, hybrid_scaler, data, steps)[-steps:]
    
    # Evaluation
    y_true = data['Close'].values[-steps:]
    
    print("\nEvaluation Results:")
    print(f"ARIMA Model - RMSE: {calculate_rmse(y_true, arima_predictions)}, MSE: {calculate_mse(y_true, arima_predictions)}, MAPE: {calculate_mape(y_true, arima_predictions)}")
    print(f"FbProphet Model - RMSE: {calculate_rmse(y_true, fbprophet_predictions)}, MSE: {calculate_mse(y_true, fbprophet_predictions)}, MAPE: {calculate_mape(y_true, fbprophet_predictions)}")
    print(f"LSTM Model - RMSE: {calculate_rmse(y_true, lstm_predictions)}, MSE: {calculate_mse(y_true, lstm_predictions)}, MAPE: {calculate_mape(y_true, lstm_predictions)}")
    print(f"ARIMA-LSTM Hybrid Model - RMSE: {calculate_rmse(y_true, hybrid_predictions)}, MSE: {calculate_mse(y_true, hybrid_predictions)}, MAPE: {calculate_mape(y_true, hybrid_predictions)}")
    
    # Plot Predictions
    plot_predictions(y_true, arima_predictions, fbprophet_predictions, lstm_predictions, hybrid_predictions, steps)
