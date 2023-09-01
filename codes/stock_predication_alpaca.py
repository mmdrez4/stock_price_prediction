"""
## Requirements
"""

import pandas as pd
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import backtrader as bt

import plotly.graph_objs as go
import mplfinance as mpf

import IPython.display as display
import seaborn as sns
import time

print("library")

length = 70

"""## Metrics

SMAPE
"""

def calculate_smape(actual, predicted):

    numerator = np.abs(predicted - actual)
    denominator = (np.abs(predicted) + np.abs(actual)) / 2

    smape = np.mean(numerator / denominator) * 100

    return smape

"""MAPE"""

def calculate_mape(actual, predicted):

    actual = np.array(actual)
    predicted = np.array(predicted)

    absolute_percentage_errors = np.abs((actual - predicted) / actual)
    mape = np.mean(absolute_percentage_errors) * 100

    return mape

"""RSME"""

def calculate_rmse(actual, predicted):
    actual = np.array(actual)
    predicted = np.array(predicted)

    squared_errors = (predicted - actual) ** 2
    mean_squared_error = np.mean(squared_errors)
    rmse = np.sqrt(mean_squared_error)

    return rmse

"""## Implementation"""

def download_stock_data(ticker, start_date, end_date, interval_time = '1h'):
    return yf.download(ticker, start=start_date, end=end_date, interval=interval_time)

def min_max_scaling(x):
    return (x - x.min()) / (x.max() - x.min())

"""### Data Preprocessing"""

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

def inverse_min_max_scaling(x, min_val, max_val):
  return x * (max_val - min_val) + min_val


"""### load Lite model"""

def predict_stock_price_quantized(X_new):
    X_new = X_new.reshape(1, X_new.shape[0], X_new.shape[1])
    X_new = X_new.astype(np.float32)
    interpreter.set_tensor(input_details[0]['index'], X_new)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])

    return output_data[0][0]

model_path = 'apple_quantized_model.tflite'
interpreter = tf.lite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print("model is loaded")

"""## Alpaca Backtrader Integration"""

"""### Backtest"""

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


class Portfolio():
  def __init__(self, cash):
    self.cash = cash
    self.volume = 0
    self.sell = []
    self.buy = []

actual_values = []

def inverse_min_max_scaling(x, min_val, max_val):
  return x * (max_val - min_val) + min_val

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
      global pf,lstm_predictions,date
      if self.counter == 0:
        predicted_price = predict_price_lstm(X)
        self.counter += 1
        lstm_predictions.append(inverse_min_max_scaling(predicted_price, x_min['Close'], x_max['Close']))
        if pf.cash > self.unit * inverse_min_max_scaling(X[-1][3], x_min['Close'], x_max['Close']) and predicted_price < X[-1][3]:
            self.buy(unit = self.unit)
            pf.volume += self.unit
            pf.cash -= self.unit * inverse_min_max_scaling(X[-1][3], x_min['Close'], x_max['Close'])
            pf.buy.append(date)
        elif pf.volume > self.unit and predicted_price > X[-1][3]:
            self.sell(unit = self.unit)
            pf.sell.append(date)
            pf.cash += self.unit * inverse_min_max_scaling(X[-1][3], x_min['Close'], x_max['Close'])
            pf.volume -= self.unit

apple_ticker = 'AAPL'
ticker = apple_ticker
start_date = '2023-05-02'
#end_date = '2023-08-02'
end_date = '2023-06-02'
historical_data = download_stock_data(ticker, start_date, end_date)
X_test, y_test= preprocess_data(historical_data)
x_min = historical_data.min()
x_max = historical_data.max()
X = X_test[0]
test_counter = 0

print("download data")

feed = bt.feeds.PandasData(dataname=historical_data)

cerebro = bt.Cerebro(stdstats=True)
cerebro.broker.setcash(10000)
cerebro.adddata(feed)
cerebro.addstrategy(LSTMStrategy, look_back=60)

pf = Portfolio(100000)
sns.set()

print("cerebro")

actual_values = []
lstm_predictions = []
historical_data.head()
test_counter = 0

###
#plt.ion()
###

for i in range(len(X_test)-1):
#for i in range(length):
  date = historical_data.index[test_counter]
  cerebro.run()
  
  
  #display.clear_output(wait=True)
  #plt.clf()
  ###
  
  
  x_vals = range(i+1)

  test_counter += 1
  X = X_test[test_counter]


  actual_values.append(inverse_min_max_scaling(X[-1][3], x_min['Close'], x_max['Close']))
  if test_counter % 10 == 0:
      print("test_counter: ", test_counter)
      plt.figure(figsize=(10, 6))
      plt.plot(x_vals, lstm_predictions,color = 'red', label='Real-Time Predictions')

      plt.plot(x_vals, actual_values,color = 'green', label='Real-Time Values')

      plt.xlabel('Time')
      plt.ylabel('Values')
      plt.title('Real-Time Prediction Visualization')
      plt.legend()
      #plt.tight_layout()
      #plt.show(block=False)
      plt.show()
  
  ###
  #time.sleep(0.1)
  ###

###
#plt.ioff()
#plt.show()
###

print("fig")
x = range(len(X_test)-1)
#x = length

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