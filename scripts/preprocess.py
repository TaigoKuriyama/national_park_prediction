import sys
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.preprocessing import LabelBinarizer

# set 'num' for number of submit files
num = sys.argv[1]

#-------------------------------------
# feature engineering for train data
#-------------------------------------

# read train and test data
input_dir = '../data/raw/'
df_train = pd.read_csv(input_dir + 'train.tsv', sep='\t')
df_test = pd.read_csv(input_dir + 'test.tsv', sep='\t')
df_test = pd.read_csv(input_dir + 'test.tsv', sep='\t')
df_holiday_exc_wkend = pd.read_csv(input_dir + 'holiday.csv')
df_nied_oyama = pd.read_csv(input_dir + 'nied_oyama.tsv', sep='\t')
df_nightley = pd.read_csv(input_dir + 'nightley.tsv', sep='\t')

# Feature engineering from datetime
def create_date_future(df):
    df['datetime'] = pd.to_datetime(df['datetime']) # dtypeをdatetime64に変換
    df['year'] = df['datetime'].dt.year
    df['month'] = df['datetime'].dt.month
    df['day'] = df['datetime'].dt.day
    df['dayofweek'] = df['datetime'].dt.dayofweek
create_date_future(df_train)
create_date_future(df_test)

# label binarize for park
def onehot_encdr_park(df):
    le = preprocessing.LabelEncoder()
    le.fit(df['park'])
    le.transform(['阿寒摩周国立公園', '十和田八幡平国立公園', '日光国立公園', '伊勢志摩国立公園','大山隠岐国立公園','阿蘇くじゅう国立公園', '霧島錦江湾国立公園', '慶良間諸島国立公園'])     
    # onehot vector for park column
    encoder = LabelBinarizer()
    park_onehot = encoder.fit_transform(df['park'])
    df_park = pd.DataFrame(park_onehot,columns=le.classes_)
    df = pd.concat([df,df_park],axis=1)
    return df
df_train = onehot_encdr_park(df_train)
df_test = onehot_encdr_park(df_test)

# label binarize for dayofweek
def onehot_encdr_dayofweek(df):
    # onehot vector for park column
    encoder = LabelBinarizer()
    park_onehot = encoder.fit_transform(df['dayofweek'])
    df_park = pd.DataFrame(park_onehot,columns=['Mon','Tue','Wed','Thu','Fri','Sat','Sun'])
    df = pd.concat([df,df_park],axis=1)
    return df
df_train = onehot_encdr_dayofweek(df_train)
df_test = onehot_encdr_dayofweek(df_test)

#-------------------------------------
# create feature
#-------------------------------------

# holiday data exclude Saturday, Sunday
df_holiday_exc_wkend['datetime'] = pd.to_datetime(df_holiday_exc_wkend['datetime'])
df_holiday_exc_wkend['dayofweek'] = df_holiday_exc_wkend['datetime'].dt.dayofweek
# dayofweekが5 or 6(土曜or日曜)の場合、holidayカラムの値を0にする
df_holiday_exc_wkend.loc[(df_holiday_exc_wkend['dayofweek'] == 5) | (df_holiday_exc_wkend['dayofweek'] == 6), 'holiday'] = 0
df_holiday_exc_wkend = df_holiday_exc_wkend[['datetime','holiday']]
df_holiday_exc_wkend = df_holiday_exc_wkend.rename(columns={'holiday': 'pub_holiday'})

# 防災科学技術研究所 ⼤⼭鏡ヶ成積雪気象観測施設における積雪気象観測データ
# 日ごとの積雪量の平均データを作成
df_nied_oyama['日時'] = pd.to_datetime(df_nied_oyama['日時'])
df_nied_oyama['日時'] = df_nied_oyama['日時'].dt.strftime('%Y-%m-%d')
df_nied_oyama_mean = df_nied_oyama.groupby('日時',as_index=False).mean()[['日時','積雪深(cm)', '気温(℃)']]
df_oyama = df_nied_oyama_mean.assign(大山隠岐国立公園=1)
df_oyama = df_oyama.rename(columns={'日時': 'datetime'})
df_oyama.head()

# sin cos curve
N = 365 # データ数
N_ = 366
n = np.arange(365)
n_ = np.arange(366)

# Not leap year
sin_cur_not_leap= np.sin(2*np.pi*n/N)
cos_cur_not_leap= np.cos(2*np.pi*n/N)
tan_cur_not_leap= np.tan(2*np.pi*n/N)

# leap year
sin_cur_leap= np.sin(2*np.pi*n_/N_)
cos_cur_leap= np.cos(2*np.pi*n_/N_)
tan_cur_leap= np.tan(2*np.pi*n_/N_)

# concat for 2015, 2016
sin_cur = np.concatenate((sin_cur_not_leap, sin_cur_leap), axis=0)
cos_cur = np.concatenate((cos_cur_not_leap, cos_cur_leap), axis=0)
tan_cur = np.concatenate((tan_cur_not_leap, tan_cur_leap), axis=0)

# to dataframe
df_sin_cur_train = pd.DataFrame(data=sin_cur,dtype='float')
df_cos_cur_train = pd.DataFrame(data=cos_cur,dtype='float')
df_tan_cur_train = pd.DataFrame(data=tan_cur,dtype='float')
df_sin_cur_test = pd.DataFrame(data=sin_cur_not_leap,dtype='float')
df_cos_cur_test = pd.DataFrame(data=cos_cur_not_leap,dtype='float')
df_tan_cur_test = pd.DataFrame(data=tan_cur_not_leap,dtype='float')

# weather data
df_weather = pd.read_csv(input_dir + 'weather.tsv', sep='\t')
df_weather['日時'] = pd.to_datetime(df_weather['年月日'])
df_weather.loc[df_weather['地点']=='鳥羽' ,'地点'] = '伊勢志摩国立公園'
df_weather.loc[df_weather['地点']=='十和田' ,'地点'] = '十和田八幡平国立公園'
df_weather.loc[df_weather['地点']=='大山' ,'地点'] = '大山隠岐国立公園'
df_weather.loc[df_weather['地点']=='渡嘉敷' ,'地点'] = '慶良間諸島国立公園'
df_weather.loc[df_weather['地点']=='日光' ,'地点'] = '日光国立公園'
df_weather.loc[df_weather['地点']=='釧路' ,'地点'] = '阿寒摩周国立公園'
df_weather.loc[df_weather['地点']=='熊本' ,'地点'] = '阿蘇くじゅう国立公園'
df_weather.loc[df_weather['地点']=='鹿児島' ,'地点'] = '霧島錦江湾国立公園'

# extract data
df_weather = df_weather[['地点','日時','最深積雪(cm)','平均気温(℃)','降水量の合計(mm)']]
df_weather = df_weather.rename(columns={'地点': 'park','日時': 'datetime',})
df_weather[['最深積雪(cm)','平均気温(℃)','降水量の合計(mm)']] = df_weather[['最深積雪(cm)','平均気温(℃)','降水量の合計(mm)']].shift(-1)

#-------------------------------------
# merge feature for train data
#-------------------------------------

# concat data
# 学習データと休日データの結合
df_merged = pd.merge(df_train,df_holiday_exc_wkend,on='datetime')

# 学習データとsin/cosカーブデータの結合
df_datetime_train = df_train[~df_train.duplicated(subset='datetime')]['datetime']
df_datetime_train = df_datetime_train.reset_index()
df_datetime_train = df_datetime_train['datetime']
df_sin_cos_tan = pd.concat([df_datetime_train,df_sin_cur_train,df_cos_cur_train,df_tan_cur_train], axis=1)
df_sin_cos_tan.columns = ['datetime', 'sin', 'cos','tan']
df_merged = pd.merge(df_merged,df_sin_cos_tan,on='datetime')

# 学習データとweatherデータの結合
df_merged = pd.merge(df_merged,df_weather,on=['park','datetime'],how='left')
df_merged = df_merged.fillna(df_merged.mean())

# 学習に寄与しない不要なカラムの削除
df_except_y = df_merged.drop(['park','visitors','datetime','day','Mon','Tue','Wed','Thu','Fri','Sat','Sun'], axis=1)
X_train = df_except_y.as_matrix() 
y_train = df_merged['visitors'].as_matrix()

#-------------------------------------
# merge feature for test data
#-------------------------------------

# merge with holiday data
df_test_merged = pd.merge(df_test,df_holiday_exc_wkend,on='datetime')

# merge with sin/cos/tan curve data
df_datetime_test = df_test[~df_test.duplicated(subset='datetime')]['datetime']
df_datetime_test = df_datetime_test.reset_index()
df_datetime_test = df_datetime_test['datetime']
df_sin_cos_test = pd.concat([df_datetime_test,df_sin_cur_test,df_cos_cur_test,df_tan_cur_test], axis=1)
df_sin_cos_test.columns =  ['datetime', 'sin', 'cos','tan']
df_test_merged = pd.merge(df_test_merged,df_sin_cos_test,on='datetime')

# merge with weather data
df_test_merged = pd.merge(df_test_merged,df_weather,on=['park','datetime'],how='left')
df_test_merged = df_test_merged.fillna(df_test_merged.mean())

# drop columns
df_test_merged = df_test_merged.drop(['park','datetime','index','day','Mon','Tue','Wed','Thu','Fri','Sat','Sun'], axis=1)
X_test = df_test_merged.as_matrix()

#-------------------------------------
# create train and test csv
#-------------------------------------
output_dir = '../data/processed/'

df_except_y.to_csv(output_dir + 'x_train_{0}.csv'.format(num),header=None)
df_merged['visitors'].to_csv(output_dir + 'y_train_{0}.csv'.format(num),header=None)
df_test_merged.to_csv(output_dir + 'test_{0}.csv'.format(num),header=None)