"""Stock Prediction"""

"""Requirements"""

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

apple_ticker = 'AAPL'
bitcoin_ticker = "BTC-USD"
msft_ticker = "MSFT"
ticker = apple_ticker

start_date = '2022-04-12'
end_date = '2023-04-01'
data = download_stock_data(ticker, start_date, end_date)
X_train, y_train = preprocess_data(data)

## for Microsoft:
# start_date = '2023-04-02'
# end_date = '2023-06-02'
# data = download_stock_data(msft_ticker, start_date, end_date)
# x_min = data.min()
# x_max = data.max()
# X_val, y_val = preprocess_data(data)
# print(X_val.shape)

## for Bitcoin:
# start_date = '2023-04-02'
# end_date = '2023-06-02'
# data = download_stock_data(bitcoin_ticker, start_date, end_date)
# x_min = data.min()
# x_max = data.max()
# X_val, y_val = preprocess_data(data)
# print(X_val.shape)

start_date = '2023-04-02'
end_date = '2023-06-02'
data = download_stock_data(apple_ticker, start_date, end_date)
x_min = data.min()
x_max = data.max()
X_val, y_val = preprocess_data(data)

def build_lstm_model(input_shape):
    model = Sequential()
    model.add(LSTM(units=60, return_sequences=True, input_shape=input_shape))
    model.add(LSTM(units=50))
    model.add(Dense(units=1))
    learning_rate = 1e-5
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer , loss='mean_squared_error')
    return model

def inverse_min_max_scaling(x, min_val, max_val):
  return x * (max_val - min_val) + min_val

"""### Model"""

class MetricsPlotCallback(Callback):
    def __init__(self, epochs):
        super(MetricsPlotCallback, self).__init__()
        self.actual = inverse_min_max_scaling(y_val, x_min['Close'],x_max['Close'])
        self.epochs = epochs
        self.validation_data = X_val
        self.mape_scores = []
        self.smape_scores = []
        self.rmse_scores = []
        self.losses = []

    def on_epoch_end(self, epoch, logs=None):
        predicted_values = self.model.predict(self.validation_data)
        predicted_values = inverse_min_max_scaling(predicted_values, x_min['Close'],x_max['Close'])
        mape = calculate_mape(self.actual, predicted_values)
        smape = calculate_smape(self.actual, predicted_values)
        rmse = calculate_rmse(self.actual, predicted_values)

        self.mape_scores.append(mape)
        self.smape_scores.append(smape)
        self.rmse_scores.append(rmse)
        self.losses.append(logs['loss'])

        print(f'Epoch {epoch+1}/{self.epochs} - MAPE: {mape:.2f}, SMAPE: {smape:.2f}, RMSE: {rmse:.2f}')


def train_lstm_model(X_train, y_train, epochs=10, batch_size=32):
    model = build_lstm_model((X_train.shape[1], X_train.shape[2]))
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, callbacks=[metrics_callback])
    return model

epochs = 4
metrics_callback = MetricsPlotCallback(epochs)
model = train_lstm_model(X_train, y_train, epochs= epochs)

plt.figure(figsize=(10, 6))
plt.plot(range(1, epochs + 1), metrics_callback.mape_scores, label='MAPE')
plt.plot(range(1, epochs + 1), metrics_callback.smape_scores, label='SMAPE')
plt.plot(range(1, epochs + 1), metrics_callback.rmse_scores, label='RMSE')
plt.plot(range(1, epochs + 1), metrics_callback.losses, label='Loss')
plt.xlabel('Epoch')
plt.ylabel('Score')
plt.title('Model Performance Metrics Over Epochs')
plt.legend()
plt.grid(True)
plt.show()

"""### save model"""

model.save('apple_model.h5')
print("Main Model is saved!")


"""## Quantization"""

annotated_layers = [
    tfmot.quantization.keras.quantize_annotate_layer(model.layers[0]),
    tfmot.quantization.keras.quantize_annotate_layer(model.layers[1])
]
quantize_model = tf.keras.Sequential(annotated_layers + model.layers[2:])

quantize_model.compile(optimizer='adam', loss='mean_squared_error')
converter = tf.lite.TFLiteConverter.from_keras_model(quantize_model)

converter.experimental_new_converter = True

converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
converter._experimental_lower_tensor_list_ops = False

"""Save quantized model to a file"""

tflite_model = converter.convert()
with open('models/apple_quantized_model.tflite', 'wb') as f:
    f.write(tflite_model)
print("Quantized Model is saved!")