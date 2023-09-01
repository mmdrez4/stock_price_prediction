"""
## Requirements
"""

import yfinance as yf
import matplotlib.pyplot as plt

import plotly.graph_objs as go
import mplfinance as mpf


def download_stock_data(ticker, start_date, end_date, interval_time = '1h'):
    return yf.download(ticker, start=start_date, end=end_date, interval=interval_time)

start_date = '2023-05-02'
end_date = '2023-08-02'

ticker_symbol = "AAPL"

data = yf.download(ticker_symbol, start=start_date, end=end_date, interval="1h")

fig = go.Figure(data=[go.Candlestick(x=data.index,
                open=data['Open'],
                high=data['High'],
                low=data['Low'],
                close=data['Close'])])

fig.update_layout(
    title=f'Candlestick Chart for {ticker_symbol}',
    xaxis_title='Date',
    yaxis_title='Price',
    xaxis_rangeslider_visible=False
)

fig.show()

data = yf.download(ticker_symbol, start=start_date, end=end_date)
mpf.plot(data, type='candle', style='charles', title=f'{ticker_symbol} Candlestick Chart', ylabel='Price')
plt.show()
