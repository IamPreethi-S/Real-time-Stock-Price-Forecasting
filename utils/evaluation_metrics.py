# from sklearn.metrics import mean_squared_error
# import numpy as np
# import matplotlib.pyplot as plt

# def calculate_rmse(y_true, y_pred):
#     return np.sqrt(mean_squared_error(y_true, y_pred))

# def calculate_mse(y_true, y_pred):
#     return mean_squared_error(y_true, y_pred)

# def calculate_mape(y_true, y_pred):
#     return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

# def plot_predictions(y_true, arima_pred, fbprophet_pred, lstm_pred, hybrid_pred, steps):
#     plt.figure(figsize=(14, 7))
    
#     plt.plot(range(len(y_true)), y_true, label='True Prices', marker='o')
#     plt.plot(range(len(y_true)), arima_pred, label='ARIMA Predictions', marker='x')
#     plt.plot(range(len(y_true)), fbprophet_pred, label='FbProphet Predictions', marker='v')
#     plt.plot(range(len(y_true)), lstm_pred, label='LSTM Predictions', marker='s')
#     plt.plot(range(len(y_true)), hybrid_pred, label='ARIMA-LSTM Predictions', marker='d')

#     plt.title('Stock Price Predictions')
#     plt.xlabel('Days')
#     plt.ylabel('Prices')
#     plt.legend()
#     plt.grid(True)
#     plt.show()


# utils/evaluation_metrics.py
# from sklearn.metrics import mean_squared_error
# import numpy as np
# import matplotlib.pyplot as plt

# def calculate_rmse(y_true, y_pred):
#     return np.sqrt(mean_squared_error(y_true, y_pred))

# def calculate_mse(y_true, y_pred):
#     return mean_squared_error(y_true, y_pred)

# def calculate_mape(y_true, y_pred):
#     return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

# def plot_predictions(y_true, arima_pred, fbprophet_pred, lstm_pred, hybrid_pred, steps):
#     plt.figure(figsize=(14, 7))
    
#     plt.plot(range(len(y_true)), y_true, label='True Prices', marker='o')
#     plt.plot(range(len(arima_pred)), arima_pred, label='ARIMA Predictions', marker='x')
#     plt.plot(range(len(fbprophet_pred)), fbprophet_pred, label='FbProphet Predictions', marker='v')
#     plt.plot(range(len(lstm_pred)), lstm_pred, label='LSTM Predictions', marker='s')
#     plt.plot(range(len(hybrid_pred)), hybrid_pred, label='ARIMA-LSTM Predictions', marker='d')

#     plt.title('Stock Price Predictions')
#     plt.xlabel('Days')
#     plt.ylabel('Prices')
#     plt.legend()
#     plt.grid(True)
#     plt.show()


# utils/evaluation_metrics.py
from sklearn.metrics import mean_squared_error
import numpy as np
import matplotlib.pyplot as plt

def calculate_rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

def calculate_mse(y_true, y_pred):
    return mean_squared_error(y_true, y_pred)

def calculate_mape(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def plot_predictions(y_true, arima_pred, fbprophet_pred, lstm_pred, hybrid_pred, period_name):
    plt.figure(figsize=(14, 7))
    
    plt.plot(range(len(y_true)), y_true, label='True Prices', marker='o')
    plt.plot(range(len(arima_pred)), arima_pred, label='ARIMA Predictions', marker='x')
    plt.plot(range(len(fbprophet_pred)), fbprophet_pred, label='FbProphet Predictions', marker='v')
    plt.plot(range(len(lstm_pred)), lstm_pred, label='LSTM Predictions', marker='s')
    plt.plot(range(len(hybrid_pred)), hybrid_pred, label='ARIMA-LSTM Predictions', marker='d')

    plt.title(f'Stock Price Predictions over {period_name} days')
    plt.xlabel('Days')
    plt.ylabel('Prices')
    plt.legend()
    plt.grid(True)
    plt.show()
