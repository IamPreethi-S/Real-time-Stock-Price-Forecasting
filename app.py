import streamlit as st
import pandas as pd
import yfinance as yf
from prophet import Prophet
import matplotlib.pyplot as plt

# Function to fetch stock data
def fetch_stock_data(ticker, start_date, end_date):
    df = yf.download(ticker, start=start_date, end=end_date)
    df.reset_index(inplace=True)
    df = df.rename(columns={'Date': 'ds', 'Close': 'y'})
    return df

# Streamlit app
st.title("Stock Price Prediction")

ticker = st.text_input("Enter stock ticker:", "AAPL")
start_date = st.date_input("Start date", pd.to_datetime("2010-01-01"))
end_date = st.date_input("End date", pd.to_datetime("2023-07-16"))

if st.button("Predict"):
    data = fetch_stock_data(ticker, start_date, end_date)
    st.write(f"Data for {ticker} from {start_date} to {end_date}")
    st.write(data.tail())

    # Train Prophet model
    model = Prophet()
    model.fit(data)

    # Create future dataframe and make predictions
    future = model.make_future_dataframe(periods=365)
    forecast = model.predict(future)

    # Display forecast data
    st.write("Forecast data")
    st.write(forecast.tail())

    # Plot the forecast
    fig1 = model.plot(forecast)
    st.pyplot(fig1)

    # Plot forecast components
    fig2 = model.plot_components(forecast)
    st.pyplot(fig2)
