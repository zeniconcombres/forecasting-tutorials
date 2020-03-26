# pylint: disable=unused-wildcard-import
# pylint: undefined-variable
# load and plot dataset
import os, pathlib, math, sklearn, keras, scipy
# import tensorflow as tf
from pandas import DataFrame as df
from pandas import datetime, read_csv, concat
from matplotlib import pyplot
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

series = read_csv('shampoo-sales.csv', 
                  header=0, # first row in data is header - 0 by default
                  parse_dates=[0], # column 0 is parsed as dates
                  index_col=0, # first column used as index
                  squeeze=None, # true changes output of single column to type series instead of df
                  date_parser=parser) # changes format of date used as index

# summarize first few rows
print(series.head())

# transform to supervised learning
X = series.values
supervised = timeseries_to_supervised(X, 3)
print(supervised.head())

train_data, test_data = X[0:-12], X[-12:] # first 2 years are used as training data, last year test

# persistence forecast model
history = [x for x in train_data]
predictions = list()
for i in range(len(test_data)):
    # make prediction
	predictions.append(history[-1])
	# observation
	history.append(test_data[i])

rmse = sqrt(mean_squared_error(test_data, predictions)) # root mean sqr error
print('RMSE: %.3f' % rmse)

# line plot
pyplot.plot(test_data)
pyplot.plot(predictions)
# pyplot.show()