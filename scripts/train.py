# preprocess.pyで作成したファイルを読み込み、モデルを学習するモジュール。
# 学習済みモデルや特徴量、クロスバリデーションの評価結果を出力する関数も定義してください。

import pandas as pd
from sklearn.cross_validation import train_test_split
import math
import numpy as np
from sklearn.metrics import classification_report, roc_auc_score, precision_recall_curve, auc, roc_curve
import xgboost as xgb
from sklearn.grid_search import GridSearchCV

# 学習データ読み込み
df_train_x = pd.read_csv('../data/preprocess/x_train.csv')
df_train_y = pd.read_csv('../data/preprocess/y_train.csv',names='visitors')

# 行列化
X_train = df_train_x.as_matrix()
y_train = df_train_y.as_matrix()

params = {"learning_rate":[0.1,0.3,0.5],
        "max_depth": [2,3,5,10],
         "subsample":[0.5,0.8,0.9,1],
         "colsample_bytree": [0.5,1.0],
         }

# モデルにインスタンス生成
model = xgb.XGBRegressor()
cv = GridSearchCV(model,params,cv=10,n_jobs=-1)

# 予測モデルを作成
cv.fit(X_train, y_train)