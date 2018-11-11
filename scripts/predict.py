# preprocess.pyで作成したテストデータ及びtrain.pyで作成したモデルを読み込み、予測結果をファイルとして出力するモジュール。



df_test_x = pd.read_csv('../data/x_test.csv')
X_test = df_test_x.as_matrix()
