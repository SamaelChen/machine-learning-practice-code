import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt
import xgboost as xgb
import re
import gc
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report

pd.options.display.max_columns = None
# ---dm9 start--- #
main_table = pd.read_csv('/home/samael/codes/48hours/dm9.csv')
main_table['recharge5.sum_money'] = main_table['recharge5.sum_money'].fillna(0)

# clean job and machine #
job_list = ['thunder', 'flame', 'frozen']
main_table.loc[-main_table['recharge5.job'].isin(
    job_list), 'recharge5.machine'] = main_table.loc[-main_table[
        'recharge5.job'].isin(job_list), 'recharge5.job']
main_table.loc[-main_table['recharge5.job'].isin(
    job_list), 'recharge5.job'] = 'Unknown'

# drop duplicated column #
main_table = main_table.drop(['shop_buy3.roleid'], 1)

# drop dept == ios and machine == NaN #
main_table = main_table.drop([16396, 164016, 272809, 382221, 438517], 0)

# fill na #
main_table['recharge5.machine'] = main_table['recharge5.machine'].fillna('PC')

# pivot table of user and dept #
tmp1 = main_table.groupby(
    ['recharge5.roleid', 'recharge5.dept'], as_index=False).count().iloc[:, 0:3]
tmp1 = pd.pivot_table(tmp1, values='recharge5.job', index=[
                      'recharge5.roleid'], columns=['recharge5.dept'],
                      fill_value=0).reset_index()
# pivot table of user and job #
tmp2 = main_table.groupby(
    ['recharge5.roleid', 'recharge5.job'], as_index=False).count().iloc[:, 0:3]
tmp2 = pd.pivot_table(tmp2, values='recharge5.dept', index='recharge5.roleid',
                      columns='recharge5.job', fill_value=0).reset_index()

main_table = main_table.drop(
    ['recharge5.dept', 'recharge5.machine', 'recharge5.job'], 1)


main_table = main_table.groupby('recharge5.roleid', as_index=False).mean()
main_table = pd.merge(left=main_table, right=tmp1, how='left',
                      left_on='recharge5.roleid', right_on='recharge5.roleid')
main_table = pd.merge(left=main_table, right=tmp2, how='left',
                      left_on='recharge5.roleid', right_on='recharge5.roleid')

# gc #
del tmp1
del tmp2
gc.collect()
# ---dm9 end--- #

# ---addexpr start--- #
addexpr = pd.read_csv('/home/samael/codes/48hours/addexpr.csv')
addexpr.loc[addexpr['num'] > 9] = 9
tmp3 = pd.pivot_table(data=addexpr, index='addexpr2.roleid', values='num',
                      columns='addexpr2.bef_num', fill_value=-1).reset_index()
main_table = pd.merge(left=main_table, right=tmp3, how='left',
                      left_on='recharge5.roleid', right_on='addexpr2.roleid')
main_table = main_table.fillna(-1)
del tmp3
gc.collect()
# ---addexpr end--- #

# ---merge tabel---#
expend = pd.read_csv('/home/samael/codes/48hours/yb_expend_pivot.csv')
main_table = pd.merge(left=main_table, right=expend, how='left',
                      left_on='recharge5.roleid', right_on='yb_expend2.roleid')

main_table = main_table.fillna(0)
income = pd.read_csv('/home/samael/codes/48hours/yb_income_pivot.csv')
main_table = pd.merge(left=main_table, right=income, how='left',
                      left_on='recharge5.roleid', right_on='yb_income2.roleid')
main_table = main_table.fillna(0)
main_table = main_table.drop('addexpr2.roleid', 1)
# replace NaN and create y #
main_table['y'] = main_table['recharge5.sum_money'].apply(
    lambda x: 0 + (x != 0))

del expend
del addexpr
del income
gc.collect()

X = main_table.drop(['recharge5.roleid', 'recharge5.sum_money'], 1)
xgb_model = xgb.XGBClassifier()
parameters = {'nthread': [4],
              'objective': ['binary:logistic'],
              'learning_rate': [0.05],
              'max_depth': [5, 6],
              'min_child_weight': [3, 4, 5, 10],
              'subsample': [0.7],
              'colsample_bytree': [0.7],
              'n_estimators': [100],
              'seed': [1234]}

clf = GridSearchCV(estimator=xgb_model, param_grid=parameters, n_jobs=5,
                   cv=StratifiedKFold(X['y'], n_folds=5,
                                      random_state=1234, shuffle=True),
                   scoring='roc_auc', verbose=2, refit=True)

clf.fit(X.iloc[:, 0:209], X['y'])
best_parameters, score, _ = max(clf.grid_scores_, key=lambda x: x[1])
print('Raw AUC Score: ', score)
y_pred = clf.predict(X.iloc[:, :209])
confusion_matrix(X['y'], y_pred)

X_train, X_test = train_test_split(X, test_size=0.33, random_state=20170620)
bdt = AdaBoostClassifier(DecisionTreeClassifier(max_depth=20,
                                                min_samples_leaf=100,
                                                max_features=150,
                                                criterion='gini',
                                                class_weight='balanced'),
                         algorithm="SAMME",
                         n_estimators=100,
                         learning_rate=1)
bdt.fit(X_train.iloc[:, :209], X_train['y'])
y_train_pred = bdt.predict(X_train.iloc[:, :209])
y_test_pred = bdt.predict(X_test.iloc[:, :209])
print(classification_report(X_train['y'], y_train_pred))
confusion_matrix(X_train['y'], y_train_pred)
print(classification_report(X_test['y'], y_test_pred))
confusion_matrix(X_test['y'], y_test_pred)
