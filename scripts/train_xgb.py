# preprocess.pyで作成したファイルを読み込み、モデルを学習するモジュール。
# 学習済みモデルや特徴量、クロスバリデーションの評価結果を出力する関数も定義してください。
import sys
import math
import pandas as pd
import xgboost as xgb
from sklearn.grid_search import GridSearchCV
from sklearn.model_selection import train_test_split
import pickle
from sklearn.metrics import mean_absolute_error

# set number of submit files
num = sys.argv[1]
input_dir = '../data/processed/'
X_train = pd.read_csv(input_dir + 'x_train_{0}.csv'.format(num), index_col=0).as_matrix()
y_train = pd.read_csv(input_dir + 'y_train_{0}.csv'.format(num), index_col=0).as_matrix()

# parametar tuning
params = {"learning_rate":[0.1,0.3,0.5],
        "max_depth": [2,3,5,10],
         "subsample":[0.5,0.8,0.9,1],
         "colsample_bytree": [0.5,1.0],
         }
model = xgb.XGBRegressor()
cv = GridSearchCV(model,params,cv=10,n_jobs=-1)

# cross validation
X_train_cv, X_test_cv, y_train_cv, y_test_cv = train_test_split(X_train, y_train,test_size=0.5,random_state=0,shuffle=False)

# train for cv
cv.fit(X_train_cv, y_train_cv)

# predict
y_test_pred = cv.predict(X_test_cv)
y_train_pred = cv.predict(X_train_cv)

# MAE
print('MAE train：{0}'.format(mean_absolute_error(y_train_cv, y_train_pred)))
print('MAE test：{0}'.format(mean_absolute_error(y_test_cv, y_test_pred)))

# train for submit
cv.fit(X_train, y_train)
reg = xgb.XGBRegressor(**cv.best_params_)
reg.fit(X_train, y_train)

# save trained model
pickle.dump(reg, open("../model/model_{0}.pkl".format(num), "wb"))