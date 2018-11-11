# 提供データを読み込み、データに前処理を施し、モデルに入力が可能な状態でファイル出力するモジュール。
# get_train_dataやget_test_dataのように、学習用と評価用を分けて、前処理を行う関数を定義してください。 

import pandas as pd
import datetime
import numpy

# train data
df_train = pd.read_csv('../data/raw/train.tsv',sep='\t')
df_train['datetime'] = pd.to_datetime(df_train['datetime']) # dtypeをdatetime64に変換
df_train['year'] = df_train['datetime'].dt.year
df_train['month'] = df_train['datetime'].dt.month
df_train['day'] = df_train['datetime'].dt.day
df_train['dayofweek'] = df_train['datetime'].dt.dayofweek

## make park column intt num
labels, uniques = pd.factorize(df_train['park'])
df_train['park'] = labels

# test data
df_test = pd.read_csv('../data/raw/test.tsv',sep='\t')
df_test['datetime'] = pd.to_datetime(df_test['datetime']) # dtypeをdatetime64に変換
df_test['year'] = df_test['datetime'].dt.year
df_test['month'] = df_test['datetime'].dt.month
df_test['day'] = df_test['datetime'].dt.day
df_test['dayofweek'] = df_test['datetime'].dt.dayofweek

# make park column intt num
labels, uniques = pd.factorize(df_test['park'])
df_test['park'] = labels

# read holiday data
df_holiday = pd.read_csv('../data/raw/holiday.csv')
df_holiday['datetime'] = pd.to_datetime(df_holiday['datetime'])

# 学習データと休日データの結合
df_train_mrgd = pd.merge(df_holiday,df_train,on='datetime')

# 学習データ作成
df_train_x = df_train_mrgd.drop(['visitors','datetime'], axis=1)

# 正解ラベルデータ作成
df_train_y = df_train_mrgd['visitors']

# テストデータと休日データの結合
df_test_mrgd = pd.merge(df_holiday,df_test,on='datetime')
# テストデータ作成
df_test_x = df_test_mrgd.drop(['datetime'], axis=1)

# to csv
df_train_x.to_csv('../data/preprocess/x_train.csv')
df_train_y.to_csv('../data/preprocess/y_train.csv')
df_test_x.to_csv('../data/preprocess/x_test.csv')