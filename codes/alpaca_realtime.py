import backtrader as bt
import pandas as pd

# import alpaca_trade_api as tradeapi
import pandas as pd
import numpy as np

import backtrader as bt
import numpy as np

# Replace with your own Alpaca API credentials
API_KEY = 'PK440XDIDQ95UVSR5FFM'
API_SECRET = '8o3JzfrA4J88QnhjbPOFnBAxPLb86RXNK3XCa8n2'
PAPER_BASE_URL = 'https://paper-api.alpaca.markets'  # For paper trading

def predict_stock_price_quantized(X_new):
    X_new = X_new.reshape(1, X_new.shape[0], X_new.shape[1])
    X_new = X_new.astype(np.float32)
    interpreter.set_tensor(input_details[0]['index'], X_new)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])

    return output_data[0][0]

def predict_price_lstm(historical_data):
    return predict_stock_price_quantized(historical_data)

def preprocess_data(data):
    print(data['Close'].max(),data['High'].values[0])
    data = data.apply(min_max_scaling)
    print(data['Close'].max(),data['High'].values[0])

    look_back = 60
    X, y = [], []
    for i in range(len(data) - look_back):
        X.append(data[['Open', 'High', 'Low', 'Close', 'Volume']].values[i:i + look_back])
        y.append(data['Close'].values[i + look_back])
    X = np.array(X)
    y = np.array(y)

    return X, y

def download_stock_data(ticker, start_date, end_date, interval_time = '1h'):
    return yf.download(ticker, start=start_date, end=end_date, interval=interval_time)

def min_max_scaling(x):
    return (x - x.min()) / (x.max() - x.min())

! pip install yfinance

import yfinance as yf
import numpy as np

class Portfolio():
  def __init__(self, cash):
    self.cash = cash
    self.volume = 0
    self.sell = []
    self.buy = []

actual_values = []

def inverse_min_max_scaling(x, min_val, max_val):
  return x * (max_val - min_val) + min_val

risk_reward = 1.001
bought_flag = False
bought_price = 0

class LSTMStrategy(bt.Strategy):
    params = (
        ("look_back", 60),
    )
    def __init__(self):
        self.lstm_predictions = []
        self.X = X
        self.counter = 0
        self.unit = 10
    def next(self):
      global pf,lstm_predictions,date,bought_flag, test_counter, risk_reward, bought_price, X
      if self.counter == 0:
          now_price = inverse_min_max_scaling(X[-1][3], x_min['Close'], x_max['Close'])
          predicted_price = predict_price_lstm(X)
          lstm_predictions.append(inverse_min_max_scaling(predicted_price, x_min['Close'], x_max['Close']))
          if not bought_flag:
            for i in range(10):
              predicted_price = predict_price_lstm(X)
              X = X[1:,:]
              X = np.append(X, [[predicted_price, predicted_price, predicted_price, predicted_price, predicted_price]],axis = 0)
          predicted_price = inverse_min_max_scaling(predicted_price, x_min['Close'], x_max['Close'])
          if not bought_flag and pf.cash > self.unit * now_price and predicted_price * self.unit > risk_reward * self.unit * now_price:
              self.buy(unit = self.unit)
              pf.volume += self.unit
              pf.cash -= self.unit * now_price
              bought_flag = True
              pf.buy.append(date)
              bought_price = now_price
          elif (test_counter == 20 and bought_flag) or bought_price < risk_reward * now_price:
              self.sell(unit = self.unit)
              pf.sell.append(date)
              bought_flag = False
              pf.cash += self.unit * now_price
              pf.volume -= self.unit
              test_counter = 0
          else:
            test_counter += 1
          self.counter += 1

import datetime

stock_symbol = "AAPL"
ticker = yf.Ticker(stock_symbol)
historical_data = ticker.history(period="1d", interval="1m").tail(61)

X_test, y_test= preprocess_data(historical_data)
x_min = historical_data.min()
x_max = historical_data.max()
X = X_test[0]
test_counter = 0

print(len(X_test))

feed = bt.feeds.PandasData(dataname=historical_data)

cerebro = bt.Cerebro(stdstats=True)
cerebro.broker.setcash(10000)
cerebro.adddata(feed)
cerebro.addstrategy(LSTMStrategy, look_back=60)

print(historical_data.index)

pf = Portfolio(100000)
import IPython.display as display
import seaborn as sns
from datetime import datetime, timedelta
import time

sns.set()  # Set the Seaborn style

actual_values = []
lstm_predictions = []
historical_data.head()
test_counter = 0
for i in range(5):
  date = historical_data.index[test_counter]

  cerebro.run()
  time.sleep(60)
  historical_data = ticker.history(period="1d", interval="1m").tail(61)

  X_test, y_test= preprocess_data(historical_data)
  x_min = historical_data.min()
  x_max = historical_data.max()
  X = X_test[0]



  actual_values.append(inverse_min_max_scaling(X_test[-1][3], x_min['Close'], x_max['Close']))

!pip install mplfinance

!pip install plotly

import plotly.graph_objs as go
ticker_symbol = "AAPL"  # Replace with your desired stock symbol


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

# Convert the datetime index to a column for mplfinance
import mplfinance as mpf
data = yf.download(ticker, start=start_date, end=end_date)
data.reset_index(inplace=True)
data = yf.download(ticker, start=start_date, end=end_date)
mpf.plot(data, type='candle', style='charles', title=f'{ticker} Candlestick Chart', ylabel='Price')
plt.show()

import matplotlib.pyplot as plt
import numpy as np


x = range(len(X_test)-1)
fig, axs = plt.subplots(1, 2, figsize=(10, 5))
axs[0].plot(x, lstm_predictions, label='predicted')
axs[0].set_title('Predicted closed values for each hour')
axs[0].set_xlabel('x')
axs[0].set_ylabel('y')
axs[0].legend()

axs[1].plot(x, actual_values, label='actual', color='orange')
axs[1].set_title('Actual closed values for each hour')
axs[1].set_xlabel('x')
axs[1].set_ylabel('y')
axs[1].legend()

plt.tight_layout()

plt.show()

import mplfinance as mpf

print("Portfolio's cash: ", pf.cash)

stock_value = pf.volume*inverse_min_max_scaling(actual_values[-1], x_min['Close'], x_max['Close'])
print("Portfolio's stocks value: ", pf.volume*inverse_min_max_scaling(actual_values[-1], x_min['Close'], x_max['Close']))

print("Portfolio's total value: ", pf.cash + stock_value)

print("Stock is bought in these dates")
for date in pf.buy:
  print(date)
print("Stock is sold in these dates")
for date in pf.sell:
  print(date)
