import numpy as np
import pandas as pd
import scipy as sp
import scipy.stats as stats
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import geopy
import time
import datetime
import xgboost as xgb
import lightgbm as lgb
from geopy.distance import vincenty
from sklearn.linear_model import LinearRegression

pd.options.display.max_columns = None
pd.options.display.max_rows = 300

# %%
train = pd.read_csv('/home/samael/github/kaggle/train.csv')
train.shape
train.head()
distance = []
for i in range(train.shape[0]):
    distance.append(vincenty((train.loc[i, 'pickup_longitude'],
                              train.loc[i, 'pickup_latitude']),
                             (train.loc[i, 'dropoff_longitude'],
                              train.loc[i, 'dropoff_latitude'])).meters)

train['distance'] = distance
train.head()
manhattan = []
for i in range(train.shape[0]):
    manhattan.append(vincenty((train.loc[i, 'pickup_longitude'],
                               train.loc[i, 'pickup_latitude']),
                              (train.loc[i, 'pickup_longitude'],
                               train.loc[i, 'dropoff_latitude'])).meters +
                     vincenty((train.loc[i, 'pickup_longitude'],
                               train.loc[i, 'pickup_latitude']),
                              (train.loc[i, 'dropoff_longitude'],
                               train.loc[i, 'pickup_latitude'])).meters)
train['manhattan'] = manhattan
train.head()
train.to_csv('/home/samael/github/kaggle/train_dist.csv', index=False)
train = train.drop('dropoff_datetime', axis=1)
train['hour'] = train['pickup_datetime'].apply(
    lambda x: datetime.datetime.strptime(x, '%Y-%m-%d %H:%M:%S').hour)
train['weekday'] = train['pickup_datetime'].apply(
    lambda x: datetime.datetime.strptime(x, '%Y-%m-%d %H:%M:%S').weekday())
