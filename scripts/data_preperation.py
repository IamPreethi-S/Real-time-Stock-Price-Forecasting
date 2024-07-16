import yfinance as yf
import pandas as pd

def fetch_stock_data(ticker, start_date, end_date):
    # Download the data from Yahoo Finance
    df = yf.download(ticker, start=start_date, end=end_date)
    
    # Reset index to handle datetime index
    df = df.reset_index()
    
    # Rename columns to match Prophet's expected format
    df = df.rename(columns={'Date': 'ds', 'Close': 'y'})
    
    # Handle missing values by forward filling
    df['y'].fillna(method='ffill', inplace=True)
    
    return df[['ds', 'y']]

if __name__ == "__main__":
    ticker = "AAPL"
    start_date = "2010-01-01"
    end_date = "2023-07-16"
    df = fetch_stock_data(ticker, start_date, end_date)
    
    # Save the prepared data to a CSV file for later use
    df.to_csv('../data/prepared_stock_data.csv', index=False)
    
    print("Data saved to data/prepared_stock_data.csv")
    print(df.head())
