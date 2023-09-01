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

"""### Load model"""

model = tf.keras.models.load_model('apple_model.h5')
print("Main Model is Loaded!")

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