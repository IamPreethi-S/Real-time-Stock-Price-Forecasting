# models/fbprophet_model.py
import pandas as pd
from prophet import Prophet
import joblib

def train_fbprophet(data):
    df = data.rename(columns={'Date': 'ds', 'Close': 'y'})
    model = Prophet()
    model.fit(df)
    joblib.dump(model, 'models/fbprophet_model.pkl')
    return model

def predict_fbprophet(model, periods):
    future = model.make_future_dataframe(periods=periods)
    forecast = model.predict(future)
    return forecast[['ds', 'yhat']].tail(periods)

if __name__ == "__main__":
    data = pd.read_csv("data/stock_data.csv")
    fbprophet_model = train_fbprophet(data)
    print(predict_fbprophet(fbprophet_model, 10))
