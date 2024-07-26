import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
import joblib
import itertools

def train_arima(data):
    p = d = q = range(0, 3)
    pdq = list(itertools.product(p, d, q))
    best_aic = float("inf")
    best_pdq = None
    best_model = None

    for param in pdq:
        try:
            model = ARIMA(data['Close'], order=param)
            model_fit = model.fit()
            if model_fit.aic < best_aic:
                best_aic = model_fit.aic
                best_pdq = param
                best_model = model_fit
        except:
            continue

    joblib.dump(best_model, 'optimizedmodels/arima_model.pkl')
    return best_model

def predict_arima(model, steps):
    return model.forecast(steps=steps)

if __name__ == "__main__":
    data = pd.read_csv("data/stock_data.csv")
    arima_model = train_arima(data)
    print(predict_arima(arima_model, 10))
