# models/lstm_model.py
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
import joblib

def preprocess_data(data, time_step=1):
    data = data['Close'].values.reshape(-1, 1)
    scaler = MinMaxScaler(feature_range=(0, 1))
    data = scaler.fit_transform(data)
    X, y = [], []
    for i in range(len(data) - time_step - 1):
        a = data[i:(i + time_step), 0]
        X.append(a)
        y.append(data[i + time_step, 0])
    return np.array(X), np.array(y), scaler

def train_lstm(data, time_step=1):
    X, y, scaler = preprocess_data(data, time_step)
    X = X.reshape(X.shape[0], X.shape[1], 1)
    
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(time_step, 1)))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dense(25))
    model.add(Dense(1))
    
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X, y, batch_size=1, epochs=1)
    
    model.save('models/lstm_model.h5')
    joblib.dump(scaler, 'models/lstm_scaler.pkl')
    return model, scaler

def predict_lstm(model, scaler, data, time_step=1):
    X, _, _ = preprocess_data(data, time_step)
    X = X.reshape(X.shape[0], X.shape[1], 1)
    predictions = model.predict(X)
    return scaler.inverse_transform(predictions)

if __name__ == "__main__":
    data = pd.read_csv("data/stock_data.csv")
    lstm_model, lstm_scaler = train_lstm(data)
    print(predict_lstm(lstm_model, lstm_scaler, data, 10))
