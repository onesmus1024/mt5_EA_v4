import MetaTrader5 as mt5
import tensorflow as tf
import pandas as pd
from pandas.plotting import scatter_matrix
import numpy as np
import matplotlib.pyplot as plt
import datetime
import pytz
from mt5_global import settings
import time
import os
import sys
import pathlib

from mt5_actions.rates import get_rates
from mt5_global.settings import symbol, timeframe,utc_from,timezone
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

model = None
history = None

#tset time zone to UTC
utc_to = datetime.datetime.now(tz=timezone)

#get rates from mt5
rates = get_rates(symbol,timeframe, utc_from, utc_to)

# create DataFrame out of the obtained data
rates_frame = pd.DataFrame(rates)

rates_frame.drop(['time'], axis=1)
print(rates_frame.head())
rates_frame.info()
#data visualization and preprocessing  

corretion_matrix =rates_frame.corr()
#time     open     high      low    close  tick_volume  spread  real_volume
#scatter_matrix(rates_frame[attributes],figsize=(12,8))
print(corretion_matrix['close'].sort_values(ascending=False))
x=rates_frame[['open','high','low','tick_volume','spread','real_volume']]
y=rates_frame['close']
#data scaling
scaler = MinMaxScaler()
scaler.fit(x)
x_scaled = scaler.transform(x)
x = pd.DataFrame(x_scaled, columns=['open','high','low','tick_volume','spread','real_volume'])
x_train_rate,x_test_rate,y_train_rate,y_test_rates = train_test_split(x,y, test_size=0.2,shuffle=False)

root_dir = os.path.join(os.curdir, "models/saved_models")


def get_run_logdir():
    run_id = symbol+"-"+time.strftime("run_%Y_%m_%d-%H_%M_%S")
    return run_id

def create_model():
    global model
    global history

    model = keras.models.Sequential([
        keras.layers.Conv1D(filters=20, kernel_size=4, strides=2, padding="valid",
                            input_shape=[None, 1]),
        keras.layers.GRU(20, return_sequences=True),
        keras.layers.GRU(20, return_sequences=True),
        keras.layers.TimeDistributed(keras.layers.Dense(1))
    ])

    model.compile(optimizer='adam', loss='mse', metrics=['mae'])

    history = model.fit(x_train_rate, y_train_rate, epochs=20,
                        validation_split=0.2, batch_size=1)
    model.save(os.path.join(root_dir, get_run_logdir()))



def plot_learning_curves(history):
    plt.plot(history.history['loss'],label='loss',color='red')
    plt.plot(history.history['val_loss'],label='val_loss',color='blue')
    plt.legend()
    #plt.gca().set_ylim(0,0.00000001)
    plt.show()
    score = model.evaluate(x_test_rate,y_test_rates)
    print(score)
if settings.Debug:
    plot_learning_curves(history)





def predict(data):
    if model is None:
        model = keras.models.load_model(os.path.join(
            root_dir,"EURUSD-run_2022_11_05-13_51_58"))
    data = np.array(data)
    data = data.reshape(1,6)
    data = scaler.transform(data)
    data = data.reshape(1,6,1)
    prediction = model.predict(data)
    return prediction




