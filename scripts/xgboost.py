import pandas as pd
from sklearn.cross_validation import train_test_split
import math
import numpy as np
from sklearn.metrics import classification_report, roc_auc_score, precision_recall_curve, auc, roc_curve
import xgboost as xgb
from sklearn.grid_search import GridSearchCV

df_train = pd.read_csv('../data/raw/train.tsv',sep='\t')
df_test = pd.read_csv('../data/raw/test.tsv',sep='\t')
df_holiday = pd.read_csv('../data/holiday.csv')

# 学習データと休日データの結合
df_merged = pd.merge(df_holiday,df_train,on='datetime')

df_except_y = df_merged.drop(['visitors','park','datetime'], axis=1)
X_train = df_except_y.as_matrix() # Whta is this?
y_train = df_merged['visitors'].as_matrix()
X_test = df_test

params = {"learning_rate":[0.1,0.3,0.5],
        "max_depth": [2,3,5,10],
         "subsample":[0.5,0.8,0.9,1],
         "colsample_bytree": [0.5,1.0],
         }

model = xgb.XGBRegressor()
#cv = GridSearchCV(model, params, cv = 10, scoring= 'roc_auc', n_jobs =-1)
cv = GridSearchCV(model, params, cv = 10, n_jobs =-1)
cv.fit(X_train, y_train)

y_train_pred = cv.predict(X_train)
y_test_pred = cv.predict(X_test)