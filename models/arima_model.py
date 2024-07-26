# models/arima_model.py
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
import joblib

def train_arima(data):
    model = ARIMA(data['Close'], order=(5,1,0))
    model_fit = model.fit()
    joblib.dump(model_fit, 'models/arima_model.pkl')
    return model_fit

def predict_arima(model, steps):
    return model.forecast(steps=steps)

if __name__ == "__main__":
    data = pd.read_csv("data/stock_data.csv")
    arima_model = train_arima(data)
    print(predict_arima(arima_model, 10))
