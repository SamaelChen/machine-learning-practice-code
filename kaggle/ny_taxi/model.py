import numpy as np
import pandas as pd
import scipy as sp
import scipy.stats as stats
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import geopy
import datetime
import time
import xgboost as xgb
import lightgbm as lgb
from geopy.distance import vincenty
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

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
train.head()
X = train[['vendor_id', 'passenger_count', 'store_and_fwd_flag',
           'distance', 'manhattan', 'hour', 'weekday']]
y = train['trip_duration']
y = np.log(y)
X['store_and_fwd_flag'] = X['store_and_fwd_flag'].apply(
    lambda x: 0 if x == 'N' else 1)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=1234)
lgb_train = lgb.Dataset(X_train, label=y_train, categorical_feature=[
                        'vendor_id', 'store_and_fwd_flag', 'hour', 'weekday'])
lgb_test = lgb.Dataset(X_test, label=y_test, categorical_feature=[
                       'vendor_id', 'store_and_fwd_flag', 'hour', 'weekday'])
# %%
param = {}
param['application'] = 'regression'
param['boosting'] = 'gbdt'
param['num_iteration'] = 3000
param['learning_rate'] = 0.01
param['num_leaves'] = 31
param['max_depth'] = 8
param['num_threads'] = 4
param['min_data_in_leaf'] = 1000
param['metric'] = 'mse'
lgb_reg = lgb.train(param, lgb_train, num_boost_round=3000,
                    valid_sets=lgb_test)
pred = lgb_reg.predict(X_test)
# %%
resid = np.exp(y_test) - np.exp(pred)
np.mean(resid)
np.std(resid)

