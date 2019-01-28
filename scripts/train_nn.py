import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import sys

class nnet:

    def __init__(self, X, random_state=3):
        self.sess = tf.Session()
        seed = random_state
        tf.set_random_seed(seed)
        np.random.seed(seed)
        self.x_data = tf.placeholder(shape=[None, X.shape[1]], dtype=tf.float32)
        self.y_target = tf.placeholder(shape=[None, 1], dtype=tf.float32)

    def init_weight(self, shape, st_dev):
        return tf.Variable(tf.random_normal(shape, stddev=st_dev))

    def init_bias(self, shape, st_dev):
        return tf.Variable(tf.random_normal(shape, stddev=st_dev))

    def fully_connected(self, input_layer, weights, biases):
        return tf.nn.relu(tf.add(tf.matmul(input_layer, weights), biases))

    def fit(self, X, y, hidden_size, batch_size=100, iter_size=200):

        x_data = self.x_data
        y_target = self.y_target

        final_output = self.build_hidden_layer(hidden_size, X.shape[1])

        self.loss = tf.reduce_mean(tf.abs(y_target - final_output))
        self.opt = tf.train.AdamOptimizer(0.05)
        self.train_step = self.opt.minimize(self.loss)

        init = tf.global_variables_initializer()
        self.sess.run(init)

        loss_vec = self.train(X, y, iter_size, batch_size)
        plt.plot(loss_vec)
        return final_output

    def predict(self, final_output, X):
        x_data = self.x_data
        return [val[0] for val in self.sess.run(final_output, feed_dict={x_data: X})]


    def train(self, X, y, iter_size, batch_size):
        loss_vec = []
        x_data = self.x_data
        y_target = self.y_target
        for i in range(iter_size):
            rand_index = np.random.choice(len(X), size=batch_size)
            rand_x = X[rand_index]
            rand_y = np.transpose([y[rand_index]])
            self.sess.run(self.train_step, feed_dict={x_data:rand_x, y_target: rand_y})
            loss_vec.append(self.sess.run(self.loss, feed_dict={x_data:rand_x, y_target: rand_y}))

            if (i+1)%25==0:
                print('Generation:'+str(i+1)+', Loss = '+str(loss_vec[-1]))

        return loss_vec


    def build_hidden_layer(self, hidden_size, col_size):
        weights = []
        biases = []
        layers = []
        tmp_size = col_size
        x_data = self.x_data
        last_layer = x_data

        for hsize in hidden_size:
            weights.append(self.init_weight(shape=[tmp_size, hsize], st_dev=10.0))
            biases.append(self.init_bias(shape=[hsize], st_dev=10.0))
            layers.append(self.fully_connected(last_layer, weights[-1], biases[-1]))
            tmp_size = hsize
            last_layer = layers[-1]

        weights.append(self.init_weight(shape=[tmp_size, 1], st_dev=10.0))
        biases.append(self.init_bias(shape=[1], st_dev=10.0))
        layers.append(self.fully_connected(last_layer, weights[-1], biases[-1]))
        final_output = layers[-1]

        return final_output


import pandas as pd
from sklearn.model_selection import train_test_split

# set number of submit files
num = sys.argv[1]
input_dir = '../data/processed/'
X = pd.read_csv(input_dir + 'x_train_{0}.csv'.format(num), index_col=0).as_matrix()
y = pd.read_csv(input_dir + 'y_train_{0}.csv'.format(num), index_col=0).as_matrix()
df_test = pd.read_csv('../data/raw/test.tsv', sep='\t')
X_train, X_test, y_train, y_test = train_test_split(X,y, random_state=0)

# traib for cv
nnetc = nnet(X_train)
model_cv = nnetc.fit(X_train, y_train.ravel(), hidden_size=[25, 10, 3], iter_size=10000)

y_test_pred = nnetc.predict(model_cv, X_test)
y_train_pred = nnetc.predict(model_cv, X_train)

# 評価
from sklearn.metrics import mean_absolute_error
print('MAE train：{0}'.format(mean_absolute_error(y_train, y_train_pred)))
print('MAE test：{0}'.format(mean_absolute_error(y_test, y_test_pred)))

# train
model = nnetc.fit(X, y.ravel(), hidden_size=[25, 10, 3], iter_size=10000)

# predict
y_pred = nnetc.predict(model, X)

# create submit file
df_submit = pd.DataFrame({
    '':df_test['index'],
    '':y_pred})

# pandas.[DataFrame or Series].where(cond,other=xxx) condがFalseの時にotherを代入
# マイナスと予測した結果を100に修正
df_submit =df_submit.where(df_submit.iloc[:, [0]] > 0, 100)
df_submit.to_csv('../submit/submit_nn_{0}.csv'.format(num))