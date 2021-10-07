
import tensorflow.compat.v1 as tf
import tensorflow as tf1
from keras.optimizers import adam
from pandas.core.computation.expressions import evaluate
from tensorflow.python.keras.optimizers import Adam
from sklearn.metrics import classification_report
import numpy as np
tf.disable_v2_behavior()

import numpy as np
class Ensemble_Model():
    def __init__(self,input_size_ss, _weights, _biases, n_hidden):
        # DoA Prediction
        # place holder
        self.data_train_ = tf.placeholder(tf.float32, shape=[None, input_size_ss])
        # self.label_ss_ = tf.placeholder(tf.float32, shape=[None, output_size_ss * SF_NUM])
        self.label_ss_ = tf.placeholder(tf.float32, shape=[None, 121])
        self.label_ss = tf.transpose(self.label_ss_)
        self.data_train = tf.transpose(self.data_train_)  # permute n_steps and batch_size
        #Reshape to prepare input to hidden activation
        self.data_train = tf.reshape(self.data_train, [-1,input_size_ss])
        self.data_train = tf.nn.relu(tf.matmul(self.data_train_,_weights['hidden']) + _biases['hidden'])
        #Split data because rnn cell needs a list of inputs for the RNN inner loop
        self.data_train = tf.split(self.data_train, 1, 0)
        # new shape: n_steps * (batch_size, n_hidden)

        # Define two stacked LSTM cells (two recurrent layers deep) with tensorflow
        # Hai cells LSTM xếp chồng lên nhau
        self.lstm_cell_1 = tf.nn.rnn_cell.BasicLSTMCell(n_hidden, forget_bias=1.0, state_is_tuple=True)
        self.lstm_cell_2 = tf.nn.rnn_cell.BasicLSTMCell(n_hidden, forget_bias=1.0, state_is_tuple=True)
        self.lstm_cells = tf.nn.rnn_cell.MultiRNNCell([self.lstm_cell_1, self.lstm_cell_2], state_is_tuple=True)
        # Get LSTM cell output
        self.outputs, self.states = tf.nn.static_rnn(self.lstm_cells, self.data_train, dtype=tf.float32)

        # Lấy ra ouput cuối cùng cho classifier
        lstm_last_output = self.outputs[-1]
        self.output_ss = tf.matmul(lstm_last_output, _weights['out']) + _biases['out']
        self.output_ss = tf.concat(self.output_ss, axis=0)
        self.output_ss = tf.transpose(self.output_ss)
        # loss and optimizer
        self.error_ss = self.label_ss - self.output_ss
        self.loss_ss = tf.reduce_mean(tf.norm(tf.square(self.error_ss), ord=1))
        # Tối ưu hóa loss với adam
        self.train_op_ss = tf.train.AdamOptimizer(
            learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-08, use_locking=False,
            name='Adam'
        ).minimize(self.loss_ss)
        # Tính accuracy
        # self.accuracy_ss,_ = tf.metrics.accuracy(labels=tf.argmax(self.label_ss, 1),
        #                                   predictions=tf.argmax(self.output_ss, 1))
        # self.total = 0
        # self.correct_pred = 0
        # self.predicted = tf.argmax(self.output_ss, 1)
        # self.total += np.size(self.label_ss,0)
        #
        # self.correct_pred += np.sum(tf.equal(tf.argmax(self.output_ss, 1), tf.argmax(self.label_ss, 1)))
        # self.accuracy_ss = 100. * self.correct_pred / self.total
        # # self.accuracy_ss = tf.reduce_mean(tf.cast(self.correct_pred, tf.float32))
        self.correct_pred = tf.equal(tf.argmax(self.output_ss, 1), tf.argmax(self.label_ss, 1))
        self.accuracy = tf.reduce_sum(tf.cast(self.correct_pred, tf.int32))
        self._acc = tf.reduce_mean(tf.cast(self.correct_pred, tf.float32))

        # self.basicLstm1 = tf.nn.rnn_cell.LSTMCell(n_hidden)
        # self.basicLstm2 = tf.nn.rnn_cell.LSTMCell(n_hidden)
        # cell = tf.nn.rnn_cell.MultiRNNCell([self.basicLstm1,self.basicLstm2])
        #
        # self.output_rnn, _ = tf.nn.dynamic_rnn(cell=cell, inputs=self.data_train, dtype=tf.float32)
        #
        # # shape of output_rnn is: [batch_size, time_step, hidden_size]
        # self.pred = tf.layers.dense(inputs=self.output_rnn, units=121)
        #
        # self.error_ss = self.label_ss - self.pred
        # self.loss_ss = tf.reduce_mean(tf.norm(tf.square(self.error_ss), ord=1))
        # # Tối ưu hóa loss với adam
        # self.train_op_ss = tf.train.AdamOptimizer(
        #     learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-08, use_locking=False,
        #     name='Adam'
        # ).minimize(self.loss_ss)