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


# <codecell>
# [2]:

###############################################################################
#load Google stock data
#start = dt.datetime(1995,1,1)
#end   = dt.date.today()
#data = pandas_datareader.data.DataReader('GOOG','yahoo',start,end)
data = pd.read_csv('googl.csv')
#data = data[['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close']]
#data = data.set_index(['Date'])
data.head()
data.shape

data.plot(y=['Close', 'Adj Close'])
data.plot(y=['Volume'])
plt.show()
# # Normalise and Prepozess the data like a boss^12

# <codecell>
# [3]:
###############################################################################
##normalise data
#data_n = stock_data_preprocessing.normalise_stock_data(data)
#data_n.head()
#data_n.shape
#
#data_n = data_n[['Adj Volume', 'Adj Close']]
# # 1,2,3 Plot Line!

#pd.DataFrame(data['Adj Close'].cumsum()).plot()

# calculate markup of adj close, this will be the predict target y
# markup = (today price - last day price) / last day price * 100
# the first day markup=0
aryMarkup = np.zeros(shape=(len(data),1))
aryMarkup.shape
data.shape

preRow = None
for i, row in data.iterrows():
#    print(i,end='\r')
    if( preRow is None ):
        preRow = row
        continue
#    print(i)
#    aryMarkup[i] = 1
#    print(row['Adj Close'])
#    print(preRow['Adj Close'])
    aryMarkup[i] = (row['Adj Close'] - preRow['Adj Close']) / preRow['Adj Close']
    preRow = row

data['markup'] = aryMarkup*100
#data['markup'].plot(figsize=(15,6))

max_volume = data['Volume'].max()
data['Adj Volume'] = data['Volume'] / max_volume * 10

# [4]:
data_n = pd.DataFrame(data[['Adj Volume', 'markup']])
data_n.shape

data_n['markup'].plot()
plt.title('markup')
plt.show()

data_n['Adj Volume'].plot()
plt.title('Adj Volume')
plt.show()

#stock_data_preprocessing.stock_plot((data_n,))
#stock_data_preprocessing.stock_plot(data_n)

# <codecell>
###############################################################################

# make training and test data set
#
# x_train.shape = (samples, seq_steps, input_features)
# y_train.shape = (samples,)
#
# x_test.shape = (samples, seq_steps, input_features)
# y_test.shape = (samples,)

# [5]:


# training data
prediction_time = 1 # predict the markup of the prediction_time day
testdatasize = 300
seq_steps = 20
testdatacut = testdatasize + seq_steps + 1

x_train = data_n[1 : -testdatacut].as_matrix()
y_train = data_n[1+prediction_time : -testdatacut+prediction_time]['markup'].as_matrix()
#y_train = data_n[prediction_time:-testdatacut  ]['Adj Close'].as_matrix()
#y_train = data_n[prediction_time:-testdatacut  ]['Normalised Close'].as_matrix()
x_train.shape
y_train.shape

# test data
x_test = data_n[-testdatacut : -prediction_time].as_matrix()
y_test = data_n[-testdatacut+prediction_time :]['markup'].as_matrix()
#y_test = data_n[prediction_time-testdatacut:  ]['Adj Close'].as_matrix()
#y_test = data_n[prediction_time-testdatacut:  ]['Normalised Close'].as_matrix()
x_test.shape
y_test.shape

# # unroll it

# [6]:


def unroll(data,sequence_length=24):
    result = []
    for index in range(len(data) - sequence_length + 1):
        result.append(data[index: index + sequence_length])
    return np.asarray(result)


x_train = unroll(x_train,seq_steps)
x_test  = unroll(x_test,seq_steps)
y_train = y_train[-x_train.shape[0]:]
y_test  = y_test[-x_test.shape[0]:]
#y_train = y_train[-x_train.shape[0]:]
#y_test  = y_test[-x_test.shape[0]:]


print("x_train", x_train.shape)
print("y_train", y_train.shape)
print("x_test", x_test.shape)
print("y_test", y_test.shape)

# <codecell>
###############################################################################
# (with Python 3.5.3, Keras 2.0.4 and Tensorflow 1.2)

# [7]:

from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.models import Sequential


# [ ]:


#Step 2 Build Model
model = Sequential()

model.add(LSTM(
    input_dim=x_train.shape[-1],
    output_dim=128,
    return_sequences=True))
#model.add(Dropout(0.15))

model.add(LSTM(
    128,
    return_sequences=False))
#model.add(Dropout(0.2))

model.add(Dense(20))
model.add(Dense(output_dim=1))
#model.add(Activation('linear'))

start = time.time()
model.compile(loss='mse', optimizer='rmsprop')
print('compilation time : {}'.format(time.time() - start))


# <codecell>
# [ ]:
###############################################################################
#Step 3 Train the model
history = model.fit(
    x_train,
    y_train,
    batch_size=256,
    nb_epoch=25,
    validation_split=0.1)

# list all data in history
print(history.history.keys())


# summarize history for loss
plt.figure()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
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
# [ ]:
###############################################################################
# predict on test data
p = model.predict(x_test)

# plot the predicted markup value and the actural markup value
plt.figure(figsize=(12,5))
plt.plot(p,color='red', label='prediction')
plt.plot(y_test,color='blue', label='y_test')
plt.legend(loc='upper left')
plt.grid(True)
plt.show()

# transform markup back into close price
pred_test_close = np.zeros(shape=(len(x_test)+seq_steps))
pred_test_close[0:seq_steps] = data['Adj Close'][-testdatacut : -testdatacut + seq_steps]
pred_test_close.shape

for i in range(len(x_test)):
    pred_test_close[seq_steps+i] = pred_test_close[seq_steps+i-1] * (1 + p[i] / 100)

# plot predict price and actural price, calculate the log value
pred_test_close_log = np.log(pred_test_close)
data_close_log = np.log(data['Adj Close'].values)

plt.figure()
plt.plot(pred_test_close_log,color='red', label='prediction')
plt.plot(data_close_log[-len(pred_test_close_log):],color='blue', label='y_test')
plt.legend(loc='upper left')
plt.show()

# <codecell>
###############################################################################
# predict on the training data
p_train = model.predict(x_train)

# plot the predicted markup value and the actural markup value
# hard to see
plt.figure(figsize=(40,5))
plt.plot(y_train,color='blue', label='y_train')
plt.plot(p_train,color='red', label='prediction')
plt.title('markup prediction')
plt.legend(loc='upper left')
plt.show()

len(p_train)


# transform markup back into close price
pred_close = np.zeros(shape=(len(x_train)+seq_steps+1))
pred_close[0:seq_steps+1] = data['Adj Close'][0:seq_steps+1]
pred_close.shape

for i in range(len(x_train)):
    pred_close[seq_steps+1+i] = pred_close[seq_steps+i] * (1 + p_train[i] / 100)

## plot predict price and actural price
#plt.figure(figsize=(52,5))
#plt.plot(pred_close[0:1000],color='red', label='prediction')
#plt.plot(data['Adj Close'][0:1000],color='blue', label='y_train')

# plot predict price and actural price, calculate the log value
pred_close_log = np.log(pred_close)
data_close_log = np.log(data['Adj Close'].values)

plt.figure(figsize=(40,5))
plt.plot(pred_close_log[0:1000],color='red', label='prediction')
plt.plot(data_close_log[0:1000],color='blue', label='y_train')
plt.title('price(log) prediction')
plt.legend(loc='upper left')
plt.show()

## plot the predicted markup value
#plt.figure(figsize=(52,5))
#plt.plot(p_train[0:1000],color='red', label='prediction')
#plt.grid(True)
#plt.show()

##Step 4 - Plot the predictions!
#predictions = lstm.predict_sequences_multiple(model, x_test, 50, 50)
##predictions = predict_sequences_multiple(model, x_test, 50, 50)
#len(predictions)
#len(predictions[0])
#lstm.plot_results_multiple(predictions, y_test, 50)

