# pylint: disable=unused-wildcard-import
# pylint: undefined-variable
# pylint: unused-variable

"""Univariate LSTM set up following
https://machinelearningmastery.com/time-series-forecasting-long-short-term-memory-network-python/"""

# load and plot dataset
import os, pathlib, math, sklearn, keras, scipy
# import tensorflow as tf
import numpy as np
from pandas import DataFrame as df
from pandas import datetime, read_csv, concat, Series
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from math import *

directory = '/Users/yilinzheng/Documents/2020/Capstone/Tutorial/'
os.chdir(directory)

# load dataset
def parser(x):
    return datetime.strptime('190'+x, '%Y-%m')

# frame a sequence as a supervised learning problem
def timeseries_to_supervised(data, lag=1):
    output_data = df(data)
    columns = [output_data.shift(i) for i in range(1, lag+1)]
    columns.append(output_data)
    output_data = concat(columns, axis=1)
    output_data.fillna(0, inplace=True)
    return output_data

# create a differenced series interval=1 is producing a difference with the previous value
def difference(dataset, interval=1):
    diff = list()
    for i in range(interval, len(dataset)):
        value = dataset[i] - dataset[i - interval]
        diff.append(value)
    return Series(diff)
 
# invert differenced value
def inverse_difference(history, yhat, interval=1):
        return yhat + history[-interval]

### Transform timeseries data to scale
# transform scale
def scale(train, test):
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaler = scaler.fit(train)
    # transform train
    train = train.reshape(train.shape[0], train.shape[1])
    scaled_train = scaler.transform(train)
    # transform test
    test = test.reshape(test.shape[0], test.shape[1])
    scaled_test = scaler.transform(test)
    return [scaler, scaled_train, scaled_test]

def scale_back(scaler,X,value):
    new_row = [x for x in X] + [value]
    array = np.array(new_row)
    array = array.reshape(1, len(array))
    inverted = scaler.inverse_transform(array)
    return inverted[0, -1]

# fit an LSTM network to training data
def fit_lstm(train, batch_size, nb_epoch, neurons):
    X, y = train[:, 0:-1], train[:, -1]
    X = X.reshape(X.shape[0], 1, X.shape[1]) # shape this into LSTM form Samples, TimeSteps, Format
    model = Sequential()
    model.add(LSTM(neurons, batch_input_shape=(batch_size, X.shape[1], X.shape[2]), stateful=True))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    for i in range(nb_epoch):
        model.fit(X, y, epochs=1, batch_size=batch_size, verbose=0, shuffle=False)
        model.reset_states()
    return model

# make a one-step forecast
def forecast_lstm(model, batch_size, X):
    X = X.reshape(1, 1, len(X))
    yhat = model.predict(X, batch_size=batch_size)
    return yhat[0,0]

series = read_csv('shampoo-sales.csv', 
                  header=0, # first row in data is header - 0 by default
                  parse_dates=[0], # column 0 is parsed as dates
                  index_col=0, # first column used as index
                  squeeze=None, # true changes output of single column to type series instead of df
                  date_parser=parser) # changes format of date used as index

# summarize first few rows
print(series.head())

# transform data to be stationary - difference between values of previous interval
raw_values = series.values
diff_values = difference(raw_values, 1)
 
# transform data to be supervised learning - essentially shifting the data down an interval
supervised = timeseries_to_supervised(diff_values, 1)
supervised_values = supervised.values
# print(supervised.head())
 
# split data into train and test-sets
train_data, test_data = supervised_values[0:-12], supervised_values[-12:]

# transform the scale of the data
scaler, train_scaled, test_scaled = scale(train_data, test_data)

# fit the model
lstm_model = fit_lstm(train_scaled, 1, 1500, 1) # Q: how to choose epochs?
# forecast the entire training dataset to build up state for forecasting
train_reshaped = train_scaled[:, 0].reshape(len(train_scaled), 1, 1)
lstm_model.predict(train_reshaped, batch_size=1)
print(len(train_data))
print(len(test_data))
print(len(train_scaled))
print(len(test_scaled))
print(len(raw_values))

# walk-forward validation on the test data
predictions = list()
for i in range(len(test_scaled)):
    # make one-step forecast
    X, y = test_scaled[i, 0:-1], test_scaled[i, -1]
    yhat = forecast_lstm(lstm_model, 1, X)
    # invert scaling
    yhat = scale_back(scaler, X, yhat)
    # invert differencing
    yhat = inverse_difference(raw_values, yhat, len(test_scaled)+1-i)
    # store forecast
    predictions.append(yhat)
    expected = raw_values[len(train_data)+i+1]
    print('Month=%d, Predicted=%f, Expected=%f' % (i+1, yhat, expected))

rmse = sqrt(mean_squared_error(raw_values[-12:], predictions)) # root mean sqr error
print('RMSE: %.3f' % rmse)

# line plot
plt.plot(raw_values[-12:])
plt.plot(predictions)
plt.show()

