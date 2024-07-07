# data_collection.py
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import StandardScaler

# Fetch historical data
def fetch_data(ticker, start_date, end_date):
    data = yf.download(ticker, start=start_date, end=end_date)
    data.dropna(inplace=True)
    return data

# Preprocess the data
def preprocess_data(data):
    data['Return'] = data['Close'].pct_change()
    data['Volatility'] = data['Close'].rolling(window=21).std()
    data.dropna(inplace=True)

    scaler = StandardScaler()
    data[['Return', 'Volatility']] = scaler.fit_transform(data[['Return', 'Volatility']])
    return data

# Example usage
ticker = 'AAPL'
start_date = '2020-01-01'
end_date = '2023-01-01'
data = fetch_data(ticker, start_date, end_date)
data = preprocess_data(data)
data.to_csv('preprocessed_data.csv', index=False)
