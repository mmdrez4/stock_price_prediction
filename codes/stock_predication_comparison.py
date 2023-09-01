"""Stock Prediction"""

"""
## Requirements
"""

import pandas as pd
import yfinance as yf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import Callback
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
import tensorflow.lite as tflite
import tensorflow_model_optimization as tfmot


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


"""## Load the quantized model"""

model_path = 'apple_quantized_model.tflite'
interpreter = tf.lite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

"""### Run inference"""

def predict_stock_price_quantized(X_new):
    X_new = X_new.reshape(1, X_new.shape[0], X_new.shape[1])
    X_new = X_new.astype(np.float32)
    interpreter.set_tensor(input_details[0]['index'], X_new)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])

    return output_data[0][0]

"""## Comparison between Models"""

"""### load plain model"""

def predict_stock_price(X_new):
    X_new = X_new.reshape(1, X_new.shape[0], X_new.shape[1])
    predicted_price = plain_model.predict(X_new)
    return predicted_price[0][0]

model_path = 'apple_model.h5'
plain_model = tf.keras.models.load_model(model_path)

"""### load Lite model model"""

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


"""### Metrics"""

ticker = 'AAPL'
start_date = '2023-05-02'
end_date = '2023-08-03'

data = download_stock_data(ticker, start_date, end_date)
x_min = data.min()
x_max = data.max()
data.shape

X_new, y_new = preprocess_data(data)

## Quantized Model
lite_predictions = []
for i in range(len(X_new)):
    prediction = predict_stock_price_quantized(X_new[i])
    lite_predictions.append(prediction)

predictions_reversed = inverse_min_max_scaling(np.asarray(lite_predictions), x_min['Close'],x_max['Close'])

y_new = np.asarray(y_new)
y_new_rev = inverse_min_max_scaling(y_new, x_min['Close'],x_max['Close'])

for i in range(len(predictions_reversed)):
    date = data.index[i]
    predicted_price = predictions_reversed[i]
    print(f"Date: {date}, Predicted Price: {predicted_price:.2f}, Actual Price: {y_new_rev[i]}")

lite_mape = calculate_mape(y_new_rev, predictions_reversed)
lite_smape = calculate_smape(y_new_rev, predictions_reversed)
lite_rmse = calculate_rmse(y_new_rev, predictions_reversed)

## Plain Model
plain_predictions = []
for i in range(len(X_new)):
    prediction = predict_stock_price(X_new[i])
    plain_predictions.append(prediction)

predictions_reversed = inverse_min_max_scaling(np.asarray(plain_predictions), x_min['Close'],x_max['Close'])

y_new = np.asarray(y_new)
y_new_rev = inverse_min_max_scaling(y_new, x_min['Close'],x_max['Close'])

for i in range(len(predictions_reversed)):
    date = data.index[i]
    predicted_price = predictions_reversed[i]
    print(f"Date: {date}, Predicted Price: {predicted_price:.2f}, Actual Price: {y_new_rev[i]}")

full_mape = calculate_mape(y_new_rev, predictions_reversed)
full_smape = calculate_smape(y_new_rev, predictions_reversed)
full_rmse = calculate_rmse(y_new_rev, predictions_reversed)

counter = 0
for x, y in zip(plain_predictions, lite_predictions):
    if counter == 10:
        break
    if x != y:
        counter += 1
        print(x, y)

print(f'Full Model --- MAPE: {full_mape:.2f}, SMAPE: {full_smape:.2f}, RMSE: {full_rmse:.2f}')
print(f'Quantized Model --- MAPE: {lite_mape:.2f}, SMAPE: {lite_smape:.2f}, RMSE: {lite_rmse:.2f}')