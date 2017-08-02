
# coding: utf-8

# In[3]:


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
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import scale
from sklearn.preprocessing import PolynomialFeatures
from sklearn.decomposition import PCA
import xgboost as xgb
import lightgbm as lgb

pd.options.display.max_columns = None
pd.options.display.max_rows = 300


# In[ ]:


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
train.head()


# In[ ]:


train = train.drop('dropoff_datetime', axis=1)
train['hour'] = train['pickup_datetime'].apply(
    lambda x: datetime.datetime.strptime(x, '%Y-%m-%d %H:%M:%S').hour)
train['weekday'] = train['pickup_datetime'].apply(
    lambda x: datetime.datetime.strptime(x, '%Y-%m-%d %H:%M:%S').weekday())


# In[ ]:


train.head()


# In[ ]:


X = train[['vendor_id', 'passenger_count', 'store_and_fwd_flag',
           'distance', 'manhattan', 'hour', 'weekday',
           'pickup_longitude', 'pickup_latitude',
           'dropoff_longitude', 'dropoff_latitude']]
y = train['trip_duration']
y = np.log(y)
X['vendor_id'] = X['vendor_id'].apply(lambda x: 'vendor_id_' + str(x))
X['store_and_fwd_flag'] = X['store_and_fwd_flag'].apply(
    lambda x: 'store_and_fwd_flag_' + x)
X['hour'] = X['hour'].apply(lambda x: 'hour_' + str(x))
X['weekday'] = X['weekday'].apply(lambda x: 'weekday_' + str(x))


# In[ ]:


X_dummies = pd.get_dummies(
    X[['vendor_id', 'store_and_fwd_flag', 'hour', 'weekday']])


# In[ ]:


X = pd.concat([X[['distance', 'manhattan',
                  'pickup_longitude', 'pickup_latitude',
                  'dropoff_longitude', 'dropoff_latitude']], X_dummies], axis=1)


# In[ ]:


coords = np.vstack((X[['pickup_latitude', 'pickup_longitude']].values,
                    X[['dropoff_latitude', 'dropoff_longitude']].values,
                    X[['pickup_latitude', 'pickup_longitude']].values,
                    X[['dropoff_latitude', 'dropoff_longitude']].values))

pca = PCA().fit(coords)


# In[ ]:


X['pickup_pca0'] = pca.transform(
    X[['pickup_latitude', 'pickup_longitude']])[:, 0]
X['pickup_pca1'] = pca.transform(
    X[['pickup_latitude', 'pickup_longitude']])[:, 1]
X['dropoff_pca0'] = pca.transform(
    X[['dropoff_latitude', 'dropoff_longitude']])[:, 0]
X['dropoff_pca1'] = pca.transform(
    X[['dropoff_latitude', 'dropoff_longitude']])[:, 1]


# In[ ]:


X['log_distance'] = np.log(X['distance'] + 1)
X['log_manhattan'] = np.log(X['manhattan'] + 1)
X['log_great_circle'] = np.log(X['great_circle'] + 1)


# In[ ]:


poly = PolynomialFeatures(degree=2)


# In[ ]:


X = poly.fit_transform(X)


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=1234)
lgb_train = lgb.Dataset(X_train, label=y_train)
lgb_test = lgb.Dataset(X_test, label=y_test)


# In[ ]:


param = {}
param['application'] = 'regression'
param['boosting'] = 'gbdt'
param['num_iteration'] = 5000
param['learning_rate'] = 0.005
param['num_leaves'] = 2000
param['max_depth'] = 20
param['num_threads'] = 16
param['min_data_in_leaf'] = 10
param['metric'] = 'rmse'
lgb_reg = lgb.train(param, lgb_train, num_boost_round=10000,
                    valid_sets=lgb_test)
pred = lgb_reg.predict(X_test)


# In[ ]:


np.sqrt(mean_squared_error(np.log(np.exp(pred) + 1),
                           np.log(np.exp(y_test) + 1)))


# In[106]:


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
val.head()


# In[107]:


val['hour'] = val['pickup_datetime'].apply(
    lambda x: datetime.datetime.strptime(x, '%Y-%m-%d %H:%M:%S').hour)
val['weekday'] = val['pickup_datetime'].apply(
    lambda x: datetime.datetime.strptime(x, '%Y-%m-%d %H:%M:%S').weekday())


# In[108]:


X_pred = val[['vendor_id', 'passenger_count', 'store_and_fwd_flag',
              'distance', 'manhattan', 'hour', 'weekday',
              'pickup_longitude', 'pickup_latitude',
              'dropoff_longitude', 'dropoff_latitude']]
X_pred['vendor_id'] = X_pred['vendor_id'].apply(
    lambda x: 'vendor_id_' + str(x))
X_pred['store_and_fwd_flag'] = X_pred['store_and_fwd_flag'].apply(
    lambda x: 'store_and_fwd_flag_' + x)
X_pred['hour'] = X_pred['hour'].apply(lambda x: 'hour_' + str(x))
X_pred['weekday'] = X_pred['weekday'].apply(lambda x: 'weekday_' + str(x))


# In[109]:


X_dummies = pd.get_dummies(
    X_pred[['vendor_id', 'store_and_fwd_flag', 'hour', 'weekday']])
X_pred = pd.concat([X_pred[['distance', 'manhattan',
                            'pickup_longitude', 'pickup_latitude',
                            'dropoff_longitude', 'dropoff_latitude']],
                    X_dummies], axis=1)
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
X_pred['great_circle'] = np.log(X_pred['great_circle'] + 1)


# In[110]:


X_pred.head()


# In[111]:


pred = lgb_reg.predict(X_pred)


# In[117]:


val['trip_duration'] = np.exp(pred)


# In[118]:


val.head()


# In[119]:


val[['id', 'trip_duration']].to_csv(
    '/home/samael/kaggle/submission.csv', index=False)


# In[ ]:
