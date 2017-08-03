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
train.shape
train.head()
distance = []
for i in range(train.shape[0]):
    distance.append(vincenty((train.loc[i, 'pickup_longitude'],
                              train.loc[i, 'pickup_latitude']),
                             (train.loc[i, 'dropoff_longitude'],
                              train.loc[i, 'dropoff_latitude'])).meters)

train['distance'] = distance
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
great_circle_dist = []
for i in range(train.shape[0]):
    great_circle_dist.append(great_circle(
        (train.loc[i, 'pickup_longitude'],
         train.loc[i, 'pickup_latitude']),
        (train.loc[i, 'dropoff_longitude'],
         train.loc[i, 'dropoff_latitude'])).meters)
train['great_circle'] = great_circle_dist
train['pickup_datetime'] = train['pickup_datetime'].apply(
    lambda x: datetime.datetime.strptime(x, '%Y-%m-%d %H:%M:%S'))
train['hour'] = train['pickup_datetime'].apply(lambda x: x.hour)
train['minute'] = train['pickup_datetime'].apply(lambda x: x.minute)
train['weekday'] = train['pickup_datetime'].apply(lambda x: x.weekday())
train['weekday_of_year'] = train[
    'pickup_datetime'].apply(lambda x: x.weekofyear)
train['speed_1st'] = train['distance'] / (train['trip_duration'] + 1)
train['speed_2nd'] = train['manhattan'] / (train['trip_duration'] + 1)
train['speed_3rd'] = train['great_circle'] / (train['trip_duration'] + 1)
X = train.drop(['id'], axis=1)
y = train['trip_duration']
y = np.log(y)
X['vendor_id'] = X['vendor_id'].apply(lambda x: str(x))
X['hour'] = X['hour'].apply(lambda x: str(x))
X['weekday'] = X['weekday'].apply(lambda x: str(x))
X_dummies = pd.get_dummies(
    X[['vendor_id', 'store_and_fwd_flag', 'hour', 'weekday']])
X = pd.concat([X.drop(['vendor_id', 'store_and_fwd_flag',
                       'hour', 'weekday'], axis=1), X_dummies], axis=1)
coords = np.vstack((X[['pickup_latitude', 'pickup_longitude']].values,
                    X[['dropoff_latitude', 'dropoff_longitude']].values,
                    X[['pickup_latitude', 'pickup_longitude']].values,
                    X[['dropoff_latitude', 'dropoff_longitude']].values))

pca = PCA().fit(coords)
X['pickup_pca0'] = pca.transform(
    X[['pickup_latitude', 'pickup_longitude']])[:, 0]
X['pickup_pca1'] = pca.transform(
    X[['pickup_latitude', 'pickup_longitude']])[:, 1]
X['dropoff_pca0'] = pca.transform(
    X[['dropoff_latitude', 'dropoff_longitude']])[:, 0]
X['dropoff_pca1'] = pca.transform(
    X[['dropoff_latitude', 'dropoff_longitude']])[:, 1]
X['log_distance'] = np.log(X['distance'] + 1)
X['log_manhattan'] = np.log(X['manhattan'] + 1)
X['log_great_circle'] = np.log(X['great_circle'] + 1)
kmeans = MiniBatchKMeans(n_clusters=100, batch_size=10000).fit(coords)
X.loc[:, 'pickup_cluster'] = kmeans.predict(
    train[['pickup_latitude', 'pickup_longitude']])
X.loc[:, 'dropoff_cluster'] = kmeans.predict(
    train[['dropoff_latitude', 'dropoff_longitude']])
pickup_mean_duration = X[['pickup_cluster', 'trip_duration']].groupby(
    'pickup_cluster').mean()
dropoff_mean_duration = X[['dropoff_cluster', 'trip_duration']].groupby(
    'dropoff_cluster').mean()
pickup_mean_speed1st = X[['pickup_cluster', 'speed_1st']].groupby(
    'pickup_cluster').mean()
pickup_mean_speed2nd = X[['pickup_cluster', 'speed_2nd']].groupby(
    'pickup_cluster').mean()
pickup_mean_speed3rd = X[['pickup_cluster', 'speed_3rd']].groupby(
    'pickup_cluster').mean()
dropoff_mean_speed1st = X[['dropoff_cluster', 'speed_1st']].groupby(
    'dropoff_cluster').mean()
dropoff_mean_speed2nd = X[['dropoff_cluster', 'speed_2nd']].groupby(
    'dropoff_cluster').mean()
dropoff_mean_speed3rd = X[['dropoff_cluster', 'speed_3rd']].groupby(
    'dropoff_cluster').mean()
X['pickup_mean_duration'] = pickup_mean_duration
X['dropoff_mean_duration'] = dropoff_mean_duration
X['pickup_mean_speed_1st'] = pickup_mean_speed1st
X['pickup_mean_speed_2nd'] = pickup_mean_speed2nd
X['pickup_mean_speed_3rd'] = pickup_mean_speed3rd
X['dropoff_mean_speed_1st'] = dropoff_mean_speed1st
X['dropoff_mean_speed_2nd'] = dropoff_mean_speed2nd
X['dropoff_mean_speed_3rd'] = dropoff_mean_speed3rd
X = X.drop(['trip_duration', 'speed_1st', 'speed_2nd', 'speed_3rd',
            'pickup_datetime', 'dropoff_datetime'], axis=1)
# poly = PolynomialFeatures(degree=2)
# X = poly.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=1234)
lgb_train = lgb.Dataset(X_train, label=y_train)
lgb_test = lgb.Dataset(X_test, label=y_test)
param = {}
param['application'] = 'regression'
param['boosting'] = 'gbdt'
param['num_iteration'] = 5000
param['learning_rate'] = 0.005
param['num_leaves'] = 2000
param['max_depth'] = 20
param['num_threads'] = 32
param['min_data_in_leaf'] = 10
param['metric'] = 'rmse'
lgb_reg = lgb.train(param, lgb_train, num_boost_round=10000,
                    valid_sets=lgb_test)

val = pd.read_csv('/home/samael/kaggle/test.csv')
distance = []
for i in range(val.shape[0]):
    distance.append(vincenty((val.loc[i, 'pickup_longitude'],
                              val.loc[i, 'pickup_latitude']),
                             (val.loc[i, 'dropoff_longitude'],
                              val.loc[i, 'dropoff_latitude'])).meters)

val['distance'] = distance
manhattan = []
for i in range(val.shape[0]):
    manhattan.append(vincenty((val.loc[i, 'pickup_longitude'],
                               val.loc[i, 'pickup_latitude']),
                              (val.loc[i, 'pickup_longitude'],
                               val.loc[i, 'dropoff_latitude'])).meters +
                     vincenty((val.loc[i, 'pickup_longitude'],
                               val.loc[i, 'pickup_latitude']),
                              (val.loc[i, 'dropoff_longitude'],
                               val.loc[i, 'pickup_latitude'])).meters)
val['manhattan'] = manhattan
great_circle_dist = []
for i in range(val.shape[0]):
    great_circle_dist.append(great_circle(
        (val.loc[i, 'pickup_longitude'],
         val.loc[i, 'pickup_latitude']),
        (val.loc[i, 'dropoff_longitude'],
         val.loc[i, 'dropoff_latitude'])).meters)
val['great_circle'] = great_circle_dist

val['pickup_datetime'] = val['pickup_datetime'].apply(
    lambda x: datetime.datetime.strptime(x, '%Y-%m-%d %H:%M:%S'))
val['hour'] = val['pickup_datetime'].apply(lambda x: x.hour)
val['minute'] = val['pickup_datetime'].apply(lambda x: x.minute)
val['weekday'] = val['pickup_datetime'].apply(lambda x: x.weekday())
val['weekday_of_year'] = val['pickup_datetime'].apply(lambda x: x.weekofyear)
X_pred = val.drop(['id'], axis=1)
X_pred['vendor_id'] = X_pred['vendor_id'].apply(lambda x: str(x))
X_pred['hour'] = X_pred['hour'].apply(lambda x: str(x))
X_pred['weekday'] = X_pred['weekday'].apply(lambda x: str(x))
X_pred_dummies = pd.get_dummies(
    X_pred[['vendor_id', 'store_and_fwd_flag', 'hour', 'weekday']])
X_pred = pd.concat([X_pred.drop(['vendor_id', 'store_and_fwd_flag',
                                 'hour', 'weekday'], axis=1),
                    X_pred_dummies], axis=1)
coords = np.vstack((X_pred[['pickup_latitude', 'pickup_longitude']].values,
                    X_pred[['dropoff_latitude', 'dropoff_longitude']].values,
                    X_pred[['pickup_latitude', 'pickup_longitude']].values,
                    X_pred[['dropoff_latitude', 'dropoff_longitude']].values))

pca = PCA().fit(coords)
X_pred['pickup_pca0'] = pca.transform(
    X_pred[['pickup_latitude', 'pickup_longitude']])[:, 0]
X_pred['pickup_pca1'] = pca.transform(
    X_pred[['pickup_latitude', 'pickup_longitude']])[:, 1]
X_pred['dropoff_pca0'] = pca.transform(
    X_pred[['dropoff_latitude', 'dropoff_longitude']])[:, 0]
X_pred['dropoff_pca1'] = pca.transform(
    X_pred[['dropoff_latitude', 'dropoff_longitude']])[:, 1]
X_pred['log_distance'] = np.log(X_pred['distance'] + 1)
X_pred['log_manhattan'] = np.log(X_pred['manhattan'] + 1)
X_pred['log_great_circle'] = np.log(X_pred['great_circle'] + 1)
kmeans = MiniBatchKMeans(n_clusters=100, batch_size=10000).fit(coords)
X_pred.loc[:, 'pickup_cluster'] = kmeans.predict(
    X_pred[['pickup_latitude', 'pickup_longitude']])
X_pred.loc[:, 'dropoff_cluster'] = kmeans.predict(
    X_pred[['dropoff_latitude', 'dropoff_longitude']])
X_pred = pd.merge(X_pred, X[['pickup_cluster', 'pickup_mean_duration']],
                  on='pickup_cluster', how='left')
X_pred = pd.merge(X_pred, X[['dropoff_cluster', 'dropoff_mean_duration']],
                  on='pickup_cluster', how='left')
X_pred = pd.merge(X_pred, X[['pickup_cluster', 'pickup_mean_speed_1st']],
                  on='pickup_cluster', how='left')
X_pred = pd.merge(X_pred, X[['pickup_cluster', 'pickup_mean_speed_2nd']],
                  on='pickup_cluster', how='left')
X_pred = pd.merge(X_pred, X[['pickup_cluster', 'pickup_mean_speed_3rd']],
                  on='pickup_cluster', how='left')
X_pred = pd.merge(X_pred, X[['dropoff_cluster', 'dropoff_mean_speed_1st']],
                  on='pickup_cluster', how='left')
X_pred = pd.merge(X_pred, X[['dropoff_cluster', 'dropoff_mean_speed_2nd']],
                  on='pickup_cluster', how='left')
X_pred = pd.merge(X_pred, X[['dropoff_cluster', 'dropoff_mean_speed_3rd']],
                  on='pickup_cluster', how='left')
pred = lgb_reg.predict(X_pred[lgb_reg.feature_name()])
val['trip_duration'] = np.exp(pred)
val[['id', 'trip_duration']].to_csv(
    '/home/samael/kaggle/submission.csv', index=False)
