import sys
sys.path.insert(0, '..')

from utils import data
import os
import sklearn
import numpy as np
from sklearn.neighbors import (
    KNeighborsClassifier,
    DistanceMetric
)
import json
from shapely.geometry import Point
import matplotlib.pyplot as plt
import geopandas as gpd
import geoplot as gplt
import mapclassify
import matplotlib.dates as mdates
import matplotlib.colors as mcolors
import pandas as pd
from pandas import read_csv
import math
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from datetime import datetime, date

def import_US_confirmed():
    BASE_PATH = './COVID-19/csse_covid_19_data/'
    MIN_CASES = 1000
    # ------------------------------------------
    
    confirmed = os.path.join(
        BASE_PATH, 
        'csse_covid_19_time_series',
        'time_series_covid19_confirmed_US.csv')
    confirmed = data.load_csv_data(confirmed)
    state_feature = {}
    states = list(set(confirmed["Province_State"]))
    # print (confirmed.iloc[232,11:confirmed.shape[0]-11])
    
    for idx in range(confirmed.shape[0]):
        if confirmed["Province_State"][idx] in list(state_feature.keys()):
            state_feature[confirmed["Province_State"][idx]] += confirmed.iloc[idx,11:confirmed.shape[0]-11]
        else:
            state_feature[confirmed["Province_State"][idx]] = confirmed.iloc[idx,11:confirmed.shape[0]-11]
    features = np.asarray(list(state_feature.values()))
    targets = np.asarray(list(state_feature.keys()))
    return features, targets

def import_US_confirmed_update():
    BASE_PATH = './COVID-19/csse_covid_19_data/'
    MIN_CASES = 1000

    confirmed = os.path.join(
        BASE_PATH, 
        'csse_covid_19_time_series',
        'us-states.csv')
    confirmed = data.load_csv_data(confirmed)
    dates = sorted(list(set(confirmed["date"])))
    _confirmed = os.path.join(
        BASE_PATH, 
        'csse_covid_19_time_series',
        'time_series_covid19_confirmed_US.csv')
    _confirmed = data.load_csv_data(_confirmed)
    state_feature = {}
    states = list(set(_confirmed["Province_State"]))
    state_feature = dict((state,np.zeros(len(dates))) for state in states)
    for idx in range(confirmed.shape[0]):
        # print ((datetime.strptime(confirmed["date"][idx], '%Y-%m-%d').date() - date(2020,1,21)).days)
        state_feature[confirmed["state"][idx]][(datetime.strptime(confirmed["date"][idx], '%Y-%m-%d').date() - date(2020,1,21)).days] =\
            confirmed["cases"][idx]
    features = np.asarray(list(state_feature.values()))
    targets = np.asarray(list(state_feature.keys()))
    return features, targets
    
    
    

def create_dataset(dataset, look_back=1):
	dataX, dataY = [], []
	for i in range(len(dataset)-look_back-1):
		a = dataset[i:(i+look_back)]
		dataX.append(a)
		dataY.append(dataset[i + look_back])
	return np.array(dataX), np.array(dataY)

def create_dataset_long_term(dataset, num_x, num_y):
    dataX, dataY, testX = [], [], []
    for i in range(len(dataset)-num_x-num_y+1):
        a = dataset[i:(i+num_x)]

        testX.append(dataset[i+num_y:i+num_x+num_y])
        dataX.append(a)
        dataY.append(dataset[i + num_x: i+num_x+num_y])
    return np.array(dataX), np.array(dataY), np.array(testX)

def LSTM_forecast(state):
    features, states = import_US_confirmed()
    features = features[np.where(states == state)[0][0]]
    trainX, trainY = create_dataset(features)
    testX, testY = create_dataset(features)
    trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
    testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))
    
    # create and fit the LSTM network
    model = Sequential()
    model.add(LSTM(32, input_shape=(1, 1), activation="relu"))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(trainX, trainY, epochs=100, batch_size=1, verbose=2)
    # make predictions
    trainPredict = model.predict(trainX)
    testPredict = model.predict(testX)
    timespan = pd.date_range("2020-01-22", periods=117).to_list()
    timespan = [str(date.date()) for date in timespan]
    # from matplotlib.dates import date2num
    # timespan =  date2num(timespan)
    fig, ax = plt.subplots()
    ax.set_xticks(np.arange(len(timespan))) 
    ax.set_xticklabels(timespan)
    # ax.xaxis.set_major_locator(mdates.YearLocator())
    # ax.xaxis.set_minor_locator(mdates.MonthLocator())
    # ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.plot(timespan, testY, label="True Cases")
    ax.plot(timespan, testPredict, label="Predicted Cases by LSTM")
    # ax.xaxis.set_major_locator(plt.MaxNLocator(5))
    plt.legend()
    plt.gcf().autofmt_xdate()

    for n, label in enumerate(ax.xaxis.get_ticklabels()):
        if n % 30 == 0:
            label.set_visible(True)
        else:
            label.set_visible(False)
    plt.show()

def LSTM_forecast_long_term(state,x_days,y_days):
    features, states = import_US_confirmed()
    features = features[np.where(states == state)[0][0]]
    features_update, states_update = import_US_confirmed_update()
    features_update = features_update[np.where(states_update == state)[0][0]]
    trainX, trainY, testX = create_dataset_long_term(features,x_days,y_days)
    
    trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
    testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))
    
    # create and fit the LSTM network
    model = Sequential()
    model.add(LSTM(1024, input_shape=(1, x_days), activation="relu"))
    model.add(Dense(y_days))
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(trainX, trainY, epochs=100, batch_size=1, verbose=2)
    # make predictions
    trainPredict = model.predict(trainX)
    testPredict = model.predict(testX)
    timespan = pd.date_range("2020-01-22", "2020-05-19").to_list()
    timespan = [str(date.date()) for date in timespan]
    timespan_update = pd.date_range("2020-01-21", "2020-06-08").to_list()
    timespan_update = [str(date.date()) for date in timespan_update]
    # print (testPredict)
    forecast_timespan = pd.date_range("2020-05-19", periods = y_days + 1).to_list()
    forecast_timespan = [str(date.date()) for date in forecast_timespan]
    fig, ax = plt.subplots()
    ax.set_xticks(np.arange(len(timespan_update))) 
    ax.set_xticklabels(timespan_update)
    
    ax.plot(timespan_update,features_update, label="True Cases 2020-05-20 ~ 2020-06-08")
    # plt.plot(range(0,x_days), testX[-1][0])
    

     
    ax.plot(forecast_timespan, np.insert(testPredict[-1], 0, features[-1]), label="Predicted Cases by LSTM 2020-05-20 ~ 2020-06-19")
    ax.plot(timespan, features, label = "Trained Cases 2020-01-21 ~ 2020-05-19")
    ax.legend()
    for n, label in enumerate(ax.xaxis.get_ticklabels()):
        if n % 30 == 0:
            label.set_visible(True)
        else:
            label.set_visible(False)


    plt.show()
    
def smooth(x,window_len=11,window='hanning'):
   

    s=np.r_[x[window_len-1:0:-1],x,x[-2:-window_len-1:-1]]
    #print(len(s))
    if window == 'flat': #moving average
        w=np.ones(window_len,'d')
    else:
        w=eval('np.'+window+'(window_len)')

    y=np.convolve(w/w.sum(),s,mode='valid')
    return y



if __name__ == "__main__":

    # BASE_PATH = './COVID-19/csse_covid_19_data/'
    # MIN_CASES = 1000
    # # ------------------------------------------
    
    # confirmed = os.path.join(
    #     BASE_PATH, 
    #     'csse_covid_19_time_series',
    #     'time_series_covid19_confirmed_US.csv')
    # confirmed = data.load_csv_data(confirmed)
    
    
    
    # features, states = import_US_confirmed()
    # features_california = features[np.where(states == "California")[0][0]]
    
    # timespan = pd.date_range("2020-01-22", "2020-05-19").to_list()
    # timespan = [str(date.date()) for date in timespan]

    # dict_state_feature = dict((states[idx], features[idx]) for idx in range(states.shape[0]))
    
    # trainX, trainY = create_dataset(features_california)
    # testX, testY = create_dataset(features_california)
    
    # # reshape input to be [samples, time steps, features]
    # trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
    # testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))
    
    # # create and fit the LSTM network
    # model = Sequential()
    # model.add(LSTM(4, input_shape=(1, 1), activation="relu"))
    # model.add(Dense(1))
    # model.compile(loss='mean_squared_error', optimizer='adam')
    # model.fit(trainX, trainY, epochs=100, batch_size=1, verbose=2)
    # # make predictions
    # trainPredict = model.predict(trainX)
    # testPredict = model.predict(testX)
    
    # plt.plot(testY, label="True Cases")
    # plt.plot(testPredict, label="Predicted Cases by LSTM")
    # plt.legend()
    # plt.show()
            
    # LSTM_forecast("Illinois")



    # trainX, trainY = create_dataset_long_term(features_california,60,30)
    # testX, testY = create_dataset_long_term(features_california,60,30)
    # trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
    # testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))
    # # create and fit the LSTM network
    # model = Sequential()
    # model.add(LSTM(1024, input_shape=(1, 60), activation="relu"))
    # model.add(Dense(30))
    # model.compile(loss='mean_squared_error', optimizer='adam')
    # model.fit(trainX, trainY, epochs=100, batch_size=1, verbose=2)
    # # make predictions
    # trainPredict = model.predict(trainX)
    # testPredict = model.predict(testX)
    
    # plt.plot(testY[-1], label="True Cases")
    # plt.plot(testPredict[-1], label="Predicted Cases by LSTM")
    # plt.legend()
    # plt.show()
    
    # features_update, states_update = import_US_confirmed_update()
    # california_update = features_update[np.where(states_update == "California")[0][0]]
    
    # timespan_update = pd.date_range("2020-01-21", "2020-06-08").to_list()
    # timespan_update = [str(date.date()) for date in timespan_update]
    
    # plt.plot(timespan_update[-60:-20], california_update[-60:-20], label="True Cases")
    # plt.plot(timespan[-30:], testPredict[-1], label = "LSTM")
    # plt.plot()
    # LSTM_forecast("California")
    LSTM_forecast_long_term("Illinois",75,30)
    
    
    