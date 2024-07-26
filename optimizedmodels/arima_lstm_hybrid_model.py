# models/arima_lstm_hybrid_model.py
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from keras.models import Sequential
from keras.layers import Dense, LSTM
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import joblib

def preprocess_data(data, time_step=1):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)
    X, y = [], []
    for i in range(len(scaled_data) - time_step - 1):
        X.append(scaled_data[i:(i + time_step), 0])
        y.append(scaled_data[i + time_step, 1])
    X = np.array(X)
    y = np.array(y)
    return X, y, scaler

def train_arima_lstm(data, time_step=1):
    # ARIMA Model
    arima_model = ARIMA(data['Close'], order=(5,1,0))
    arima_model_fit = arima_model.fit()
    
    # LSTM Model
    data['ARIMA'] = arima_model_fit.predict(start=0, end=len(data)-1)
    X, y, scaler = preprocess_data(data[['ARIMA', 'Close']], time_step)
    X = X.reshape(X.shape[0], X.shape[1], 1)
    
    lstm_model = Sequential()
    lstm_model.add(LSTM(50, return_sequences=True, input_shape=(time_step, 1)))
    lstm_model.add(LSTM(50, return_sequences=False))
    lstm_model.add(Dense(25))
    lstm_model.add(Dense(1))
    
    lstm_model.compile(optimizer='adam', loss='mean_squared_error')
    lstm_model.fit(X, y, batch_size=1, epochs=1)
    
    joblib.dump(arima_model_fit, 'optimizedmodels/arima_model.pkl')
    lstm_model.save('optimizedmodels/lstm_model.h5')
    joblib.dump(scaler, 'optimizedmodels/lstm_scaler.pkl')
    return arima_model_fit, lstm_model, scaler

def predict_arima_lstm(arima_model, lstm_model, scaler, data, steps=1, time_step=1):
    arima_predictions = arima_model.forecast(steps=steps)
    data['ARIMA'] = arima_model.predict(start=0, end=len(data)-1)
    X, _, _ = preprocess_data(data[['ARIMA', 'Close']], time_step)
    X = X.reshape(X.shape[0], X.shape[1], 1)
    lstm_predictions = lstm_model.predict(X)
    # To ensure correct shape for inverse_transform, add dummy dimension and remove after transformation
    lstm_predictions = lstm_predictions.reshape(-1, 1)
    combined_data = np.hstack((lstm_predictions, lstm_predictions))
    final_predictions = scaler.inverse_transform(combined_data)[:, 0]
    return final_predictions

if __name__ == "__main__":
    data = pd.read_csv("data/stock_data.csv")
    arima_model, lstm_model, scaler = train_arima_lstm(data)
    print(predict_arima_lstm(arima_model, lstm_model, scaler, data, 10))
