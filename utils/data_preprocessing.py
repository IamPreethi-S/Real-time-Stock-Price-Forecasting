# utils/data_preprocessing.py
import yfinance as yf
import pandas as pd

def get_stock_data(ticker, start_date, end_date):
    data = yf.download(ticker, start=start_date, end=end_date)
    data.reset_index(inplace=True)
    return data

def save_data_to_csv(data, filepath):
    data.to_csv(filepath, index=False)

if __name__ == "__main__":
    ticker = "AAPL"
    start_date = "2015-01-01"
    end_date = "2023-01-01"
    data = get_stock_data(ticker, start_date, end_date)
    save_data_to_csv(data, "data/stock_data.csv")
