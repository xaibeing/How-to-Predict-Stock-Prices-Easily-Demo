# <markdowncell>
# (with Python 3.5.3, Keras 2.0.4 and Tensorflow 1.2)
#
# 2017-6-15
# input data scale to (-1,1)

# <codecell>

#get_ipython().magic('matplotlib notebook')
#get_ipython().magic('matplotlib inline')

#import os
import time
#import datetime as dt
import numpy as np
#from numpy import newaxis
import matplotlib.pyplot as plt
import pandas as pd
#import pandas_datareader
#import stock_data_preprocessing_x

from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler

# <codecell>

#load Google stock data
#start = dt.datetime(1995,1,1)
#end   = dt.date.today()
#data = pandas_datareader.data.DataReader('GOOG','yahoo',start,end)
data = pd.read_csv('googl.csv')
data.head()
data.tail()
data.shape

data.plot(y=['Close', 'Adj Close'])
data.plot(y=['Volume'])
plt.show()
# # Normalise and Prepozess the data like a boss^12

# <codecell>

data['last close'] = data['Adj Close'].shift(1)
data[['log close', 'log last close']] = np.log(data[['Adj Close', 'last close']])
data['dif'] = data['log close'] - data['log last close']

data['dif'].plot()
data['dif'].hist(bins=50)

data['limit dif'] = data['dif']

dif_limit = 0.05
data.loc[ data['limit dif'] > dif_limit, 'limit dif' ] = dif_limit
data.loc[ data['limit dif'] < -dif_limit, 'limit dif' ] = -dif_limit
data['limit dif'].plot()
data['limit dif'].hist(bins=50)

data['target dif'] = data['limit dif'].shift(-1)

#data['target close'] = data['Adj Close'].shift(-1)
#data[['log close', 'log target close']] = np.log(data[['Adj Close', 'target close']])
#data['target dif'] = data['log target close'] - data['log close']
#data['dif'] = data['Adj Close'].shift(-1)
data_sample = data[['limit dif', 'target dif']]
data_sample = data_sample.iloc[1:-1]

#data_sample['target dif'].plot()


## invert differenced value
#def inverse_difference(history, yhat, interval=1):
#	return yhat + history[-interval]

# <codecell>
test_data_size = 200
train_data = data_sample.iloc[:-test_data_size]
test_data = data_sample.iloc[-test_data_size:]
print('train_data.shape', train_data.shape)
print('test_data.shape', test_data.shape)

# <codecell>

#pd.DataFrame(data['Adj Close'].cumsum()).plot()

# scale train and test data to [-1, 1]
def scale(train, test):
	# fit scaler
	scaler = MinMaxScaler(feature_range=(-1, 1))
	scaler = scaler.fit(train)
	# transform train
#	train = train.reshape(train.shape[0], train.shape[1])
	train_scaled = scaler.transform(train)
	# transform test
#	test = test.reshape(test.shape[0], test.shape[1])
	test_scaled = scaler.transform(test)
	return scaler, train_scaled, test_scaled

## scale train and test data to [-1, 1]
#def scaleX(train, test):
#	# fit scaler
#	scaler = MinMaxScaler(feature_range=(-1, 1))
#	scaler = scaler.fit(train)
#	# transform train
##	train = train.reshape(train.shape[0], train.shape[1])
#	train_scaled = scaler.transform(train)
#	# transform test
##	test = test.reshape(test.shape[0], test.shape[1])
#	test_scaled = scaler.transform(test)
#	return scaler, train_scaled, test_scaled

## inverse scaling for a forecasted value
#def invert_scale(scaler, X):
#	new_row = [x for x in X] + [value]
#	array = np.array(new_row)
#	array = array.reshape(1, len(array))
#	inverted = scaler.inverse_transform(array)
#	return inverted[0, -1]

#def invert_scale(scaler, X, value):
#	new_row = [x for x in X] + [value]
#	array = np.array(new_row)
#	array = array.reshape(1, len(array))
#	inverted = scaler.inverse_transform(array)
#	return inverted[0, -1]

scaler_x, train_x, test_x = scale(train_data.values[:,:-1].reshape(-1,1), test_data.values[:,:-1].reshape(-1,1))
scaler_y, train_y, test_y = scale(train_data.values[:,-1].reshape(-1,1), test_data.values[:,-1].reshape(-1,1))

#scaler, train_scaled, test_scaled = scale(train_data, test_data)
train_x.shape
np.max(train_x, 0)
np.min(train_x, 0)

#plt.plot(train_x)
#plt.plot(train_y)

# <codecell>

# (with Python 3.5.3, Keras 2.0.4 and Tensorflow 1.2)

from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.models import Sequential

batch_size = 1
seq_len = 1
num_features = 1

model = Sequential()

model.add(LSTM(
    64,
    batch_input_shape=(batch_size, seq_len, num_features),
#    activation='relu',
    stateful=True,
    return_sequences=False))
#model.add(Dropout(0.3))

#model.add(LSTM(
#    128,
##    activation='relu',
#    stateful=True,
#    return_sequences=False))
#model.add(Dropout(0.2))

#model.add(Dense(20))
model.add(Dense(1))
#model.add(Dense(output_dim=1, activation='relu'))
#model.add(Activation('linear'))

start = time.time()
model.compile(loss='mse', optimizer='rmsprop')
print('compilation time : {}'.format(time.time() - start))
#train_scaled[i,0].reshape(1,1)
#train_x.shape

# <codecell>

epochs = 1
lstLoss = []
for e in range(epochs):
    for i in range(len(train_data)):
        if i >= 10:
            break;
        X = train_x[i].reshape(batch_size, seq_len, num_features)
        y = train_y[i].reshape(batch_size, 1)
        history = model.fit(
            X,
            y,
            batch_size=batch_size,
            epochs=1,
            shuffle=False,
            verbose=0)
        lstLoss.append(history.history['loss'])
    print('epoch', e, 'training loss', history.history['loss'])
    model.reset_states()

# list all data in history
#print(history.history.keys())


# summarize history for loss
plt.figure(figsize=(12,5))
plt.plot(lstLoss)
#plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
#plt.legend(['train', 'test'], loc='upper left')
plt.show()

## summarize history for accuracy
#plt.figure()
#plt.plot(history.history['acc'])
#plt.plot(history.history['val_acc'])
#plt.title('model accuracy')
#plt.ylabel('accuracy')
#plt.xlabel('epoch')
#plt.legend(['train', 'test'], loc='upper left')

# <codecell>

# predict on test data

# make a one-step forecast
def forecast_lstm(model, batch_size, X):
	X = X.reshape(1, 1, len(X))
	yhat = model.predict(X, batch_size=batch_size)
	return yhat[0,0]

# first go through the tran_data to prepare the model state
train_reshaped = train_x.reshape(len(train_x), 1, 1)
train_pred = model.predict(train_reshaped, batch_size=1)

# walk-forward validation on the test data
test_pred = list()
for i in range(len(test_x)):
	# make one-step forecast
	X, y = test_x[i, :], test_y[i]
	yhat = forecast_lstm(model, 1, X)
	# invert scaling
#	yhat = invert_scale(scaler, X, yhat)
	# invert differencing
#	yhat = inverse_difference(raw_values, yhat, len(test_scaled)+1-i)
	# store forecast
	test_pred.append(yhat)
#	expected = raw_values[len(train) + i + 1]
#	print('Predicted=%f, Expected=%f' % (i+1, yhat, expected))

# summarize history for loss
plt.figure(figsize=(12,5))
plt.plot(test_y)
plt.plot(test_pred)
#plt.plot(history.history['val_loss'])
plt.title('test_pred')
plt.show()

data.head()
plt.figure(figsize=(12,5))
plt.plot(data['target dif'][-test_data_size:])
plt.title('target dif')
plt.show()

# <codecell>

# convert prediction back to price
#test_pred_array = np.array(test_pred)
#test_pred_array = test_pred_array.reshape(len(test_pred_array), 1)
#test_pred_array.shape

#test_scaled.shape
#test_pred_array = np.zeros(shape=(len(test_pred_array), 2))
#test_pred_array.shape
#test_pred_array[:,1].shape = np.array(test_pred)
#scaler

inverted = scaler_y.inverse_transform(np.array(test_pred).reshape(len(test_pred), 1))
inverted_full = np.zeros(shape=len(data))
inverted_full[-test_data_size:] = inverted[:,0]

data['pred dif'] = inverted_full
data.shape
inverted.shape
data.tail()

data['pred log close'] = data['log close'] + data['pred dif']
data = data.append(data.iloc[-1:], ignore_index=True)
#data = data.append(pd.DataFrame(np.zeros(shape=(1, len(data.columns))), columns=data.columns), ignore_index=True)
data['pred log close'] = data['pred log close'].shift(1)
data['pred close'] = np.exp(data['pred log close'])

data.iloc[-200:].plot(y=['Adj Close', 'pred close'], figsize=(12,5))
data.iloc[-200:].plot(y=['log close', 'pred log close'], figsize=(12,5))
data.iloc[-200:].plot(y=['dif', 'pred dif'], figsize=(12,5))

data.iloc[-200:]['pred dif'].max()
data.iloc[-200:]['pred dif'].min()

# <codecell>

data['scale pred dif'] = data['pred dif'] * 20 - (np.array(test_pred).mean()-data['limit dif'].mean())
data.iloc[-200:].plot(y=['limit dif', 'scale pred dif'], figsize=(12,5))

data['pred log close'] = data['log close'] + data['scale pred dif']
data = data.append(data.iloc[-1:], ignore_index=True)
data['pred log close'] = data['pred log close'].shift(1)
data['pred close'] = np.exp(data['pred log close'])

#data.iloc[-200:].plot(y=['Adj Close', 'pred close'], figsize=(12,5))
data.iloc[-200:].plot(y=['log close', 'pred log close'], figsize=(12,5))


# <codecell>

test_pred_scale = np.array(test_pred) * 20 - 0.3

# summarize history for loss
plt.figure(figsize=(12,5))
plt.plot(test_y)
plt.plot(test_pred_scale)
#plt.plot(history.history['val_loss'])
plt.title('test_pred')
plt.show()

# <codecell>
#
## predict on the training data
#p_train = model.predict(x_train)
#
## plot the predicted markup value and the actural markup value
## hard to see
#plt.figure(figsize=(40,5))
#plt.plot(y_train,color='blue', label='y_train')
#plt.plot(p_train,color='red', label='prediction')
#plt.title('markup prediction')
#plt.legend(loc='upper left')
#plt.show()
#
#len(p_train)
#
#
## transform markup back into close price
#pred_close = np.zeros(shape=(len(x_train)+seq_steps+1))
#pred_close[0:seq_steps+1] = data['Adj Close'][0:seq_steps+1]
#pred_close.shape
#
#for i in range(len(x_train)):
#    pred_close[seq_steps+1+i] = pred_close[seq_steps+i] * (1 + p_train[i] / 100 * 0.5)
#
### plot predict price and actural price
##plt.figure(figsize=(52,5))
##plt.plot(pred_close[0:1000],color='red', label='prediction')
##plt.plot(data['Adj Close'][0:1000],color='blue', label='y_train')
#
## plot predict price and actural price, calculate the log value
#pred_close_log = np.log(pred_close)
#data_close_log = np.log(data['Adj Close'].values)
#
#plt.figure(figsize=(40,5))
#plt.plot(pred_close_log[0:1000],color='red', label='prediction')
#plt.plot(data_close_log[0:1000],color='blue', label='y_train')
#plt.title('price(log) prediction')
#plt.legend(loc='upper left')
#plt.show()
#
### plot the predicted markup value
##plt.figure(figsize=(52,5))
##plt.plot(p_train[0:1000],color='red', label='prediction')
##plt.grid(True)
##plt.show()
#
###Step 4 - Plot the predictions!
##predictions = lstm.predict_sequences_multiple(model, x_test, 50, 50)
###predictions = predict_sequences_multiple(model, x_test, 50, 50)
##len(predictions)
##len(predictions[0])
##lstm.plot_results_multiple(predictions, y_test, 50)

