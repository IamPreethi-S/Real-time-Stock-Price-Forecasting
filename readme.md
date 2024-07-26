# Real-time Stock Price Prediction

This project provides a real-time stock price prediction dashboard using Streamlit and the Prophet forecasting model. It fetches stock data from Yahoo Finance and generates future stock price forecasts.


## Steps to Run

1. **Clone the repository**:
    git clone https://github.com/IamPreethi-S/Real-time-Stock-Price-Forecasting

2. **Set up the virtual environment**:

3. **Install dependencies**:
    ```
    pip install -r requirements.txt
    ```
4. **Generate forecasts**:
    python generate_forecasts.py

5. **Evaluate forecasts**:
    python evaluate_forecasts.py

6. **Visualizes the Stock Price Prediction**:
    streamlit run app.py


## Features

- **Real-time Stock Price Prediction**: Input stock tickers and date ranges to fetch historical data and generate forecasts.
- **Interactive Dashboard**: Visualize historical data, forecast data, and model components using Streamlit.
- **Data Fetching**: Integrates with Yahoo Finance to fetch up-to-date stock data.
- **Model Training and Evaluation**: Used ARIMA, LSTM, Facebook Prophet, and ARIMA-LSTM hybrid model for time series forecasting and provides metrics such as RMSE, MAE, and MAPE.

Images:

![image](https://github.com/user-attachments/assets/743d40c7-f736-4fc9-a4ca-d8217383d457)
![image](https://github.com/user-attachments/assets/1e4929af-258d-4c24-af04-a814ff2e9552)
![image](https://github.com/user-attachments/assets/d5281acf-2346-4b8b-aafb-2244aa7e3592)
![image](https://github.com/user-attachments/assets/f8ccbb79-ade4-4519-92c8-57cba48acdbc)

## Comparison of Models for Stock Price Prediction:

![image](https://github.com/user-attachments/assets/a54b1745-f5cb-4739-be91-51197030a9a2)


## Result
The ARIMA-LSTM Hybrid Model outperformed other models because it effectively combines the strengths of both linear (ARIMA) and non-linear (LSTM) modeling techniques. This dual approach allows it to handle the complex, volatile nature of stock price data more accurately, resulting in lower error metrics and better overall performance




