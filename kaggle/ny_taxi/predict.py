import numpy as np
import pandas as pd
import scipy as sp
import scipy.stats as stats
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import geopy
import time
import datetime
from geopy.distance import vincenty
from geopy.distance import great_circle
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import scale
from sklearn.preprocessing import PolynomialFeatures
from sklearn.decomposition import PCA
import xgboost as xgb
import lightgbm as lgb

pd.options.display.max_columns = None
pd.options.display.max_rows = 300

train = pd.read_csv('/home/samael/kaggle/train.csv')
y = train['trip_duration']
y = np.log(y)
X = pd.read_pickle('/home/samael/kaggle/X.pkl')
X_pred = pd.read_pickle('/home/samael/kaggle/X_pred.pkl')
val = pd.read_csv('/home/samael/kaggle/test.csv')
fastest_route_test = pd.read_csv('/home/samael/kaggle/fastest_routes_test.csv')
fastest_route_test = fastest_route_test[
    ['id', 'total_distance', 'total_travel_time', 'number_of_steps']]
fastest_routes_train_part_1 = pd.read_csv(
    '/home/samael/kaggle/fastest_routes_train_part_1.csv',
    usecols=['id', 'total_distance', 'total_travel_time', 'number_of_steps'])
fastest_routes_train_part_2 = pd.read_csv(
    '/home/samael/kaggle/fastest_routes_train_part_2.csv',
    usecols=['id', 'total_distance', 'total_travel_time', 'number_of_steps'])
second_fastest_routes_test = pd.read_csv(
    '/home/samael/kaggle/second_fastest_routes_test.csv',
    usecols=['id', 'total_distance', 'total_travel_time', 'number_of_steps'])
second_fastest_routes_train = pd.read_csv(
    '/home/samael/kaggle/second_fastest_routes_train.csv',
    usecols=['id', 'total_distance', 'total_travel_time', 'number_of_steps'])

X['id'] = train['id']
X_pred['id'] = val['id']
fastest_routes_train = pd.concat(
    [fastest_routes_train_part_1,
     fastest_routes_train_part_2], ignore_index=True)

X = pd.merge(X, fastest_routes_train, on='id', how='left')
X_pred = pd.merge(X_pred, fastest_route_test, on='id', how='left')

second_fastest_routes_test.head()

X = X.drop('id', axis=1)
X_pred = X_pred.drop('id', axis=1)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=1234)

lgb_train = lgb.Dataset(X_train, label=y_train)
lgb_test = lgb.Dataset(X_test, label=y_test)
param = {}
param['application'] = 'regression'
param['boosting'] = 'gbdt'
param['num_iteration'] = 5000
param['learning_rate'] = 0.05
param['num_leaves'] = 10000
param['max_depth'] = 20
param['num_threads'] = 32
param['min_data_in_leaf'] = 10
param['metric'] = 'rmse'
watchlist = [lgb_test, lgb_train]
lgb_reg = lgb.train(param, lgb_train, num_boost_round=300,
                    valid_sets=watchlist)

pred = lgb_reg.predict(X_test)
np.sqrt(mean_squared_error(np.log(np.exp(pred) + 1),
                           np.log(np.exp(y_test) + 1)))

pred = lgb_reg.predict(X_pred[lgb_reg.feature_name()])
val['trip_duration'] = np.exp(pred)
val[['id', 'trip_duration']].to_csv(
    '/home/samael/kaggle/submission.csv', index=False)

xgb_train = xgb.DMatrix(X_train, label=y_train)
xgb_test = xgb.DMatrix(X_test, label=y_test)

param = {}
param['objective'] = 'reg:linear'
param['silent'] = 0
param['booster'] = 'gbtree'
param['n_estimators'] = 5000
param['learning_rate'] = 0.1
param['max_depth'] = 30
param['nthread'] = 32
param['min_child_weight'] = 1
param['eval_metric'] = 'rmse'
watchlist = [(xgb_test, 'eval'), (xgb_train, 'train')]
xgb_reg = xgb.train(param, xgb_train, num_boost_round=150, evals=watchlist,
                    verbose_eval=1)

pred = xgb_reg.predict(xgb_test)
np.sqrt(mean_squared_error(np.log(np.exp(pred) + 1),
                           np.log(np.exp(y_test) + 1)))

pred = xgb_reg.predict(xgb.DMatrix(X_pred[X.columns.values]))
val['trip_duration'] = np.exp(pred)
val[['id', 'trip_duration']].to_csv(
    '/home/samael/kaggle/submission.csv', index=False)
