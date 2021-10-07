import matplotlib.pyplot as plt
import scipy.linalg as la
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from pandas.core.computation.expressions import evaluate
from sklearn.metrics import classification_report
import numpy as np


# function generate data
# data/batch_size
def generate_spec_batches(data_train, batch_size, noise_flag):
    if noise_flag == 0:
        data_ = data_train['input_nf']
    else:
        data_ = data_train['input']
    label_ = data_train['target_spec']
    data_len = len(label_)

    # shuffle data
    shuffle_seq = np.random.permutation(range(data_len))
    data = [data_[idx] for idx in shuffle_seq]
    label = [label_[idx] for idx in shuffle_seq]

    # generate batches
    num_batch = int(data_len / batch_size)
    data_batches = []
    label_batches = []
    for batch_idx in range(num_batch):
        batch_start = batch_idx * batch_size
        batch_end = np.min([(batch_idx + 1) * batch_size, data_len])
        data_batch = data[batch_start: batch_end]
        label_batch = label[batch_start: batch_end]
        data_batches.append(data_batch)
        label_batches.append(label_batch)

    return data_batches, label_batches


# Label one-hot
def generate_target_spectrum(DOA, doa_min, grid, NUM_GRID):
    K = len(DOA)
    target_vector = 0
    for ki in range(K):
        doa_i = DOA[ki]
        target_vector_i_ = []
        grid_idx = 0
        while grid_idx < NUM_GRID:
            # Góc trước đó
            grid_pre = doa_min + grid * grid_idx
            # Vị trí góc
            grid_post = doa_min + grid * (grid_idx + 1)
            if grid_pre <= doa_i and grid_post > doa_i:
                expand_vec = np.array([grid_post - doa_i, doa_i - grid_pre]) / grid
                # print(expand_vec)
                # # grid_idx += 2
                grid_idx += 2
            else:
                expand_vec = np.array([0.0])
                # print(expand_vec)
                grid_idx += 1
            target_vector_i_.extend(expand_vec)
        if len(target_vector_i_) >= NUM_GRID:
            target_vector_i = target_vector_i_[:NUM_GRID]

        else:
            expand_vec = np.zeros(NUM_GRID - len(target_vector_i_))
            target_vector_i = target_vector_i_
            target_vector_i.extend(expand_vec)
        target_vector += np.asarray(target_vector_i)
    # target_vector /= K
    return target_vector


def generate_training_data_ss_AI(M, N, K, d, wavelength, SNR, doa_min, doa_max, step, doa_delta, NUM_REPEAT_SS, grid_ss,
                                 NUM_GRID_SS):
    data_train_ss = {}
    data_train_ss['input_nf'] = []
    data_train_ss['input'] = []
    data_train_ss['target_spec'] = []
    for delta_idx in range(len(doa_delta)):
        delta_curr = doa_delta[delta_idx]  # inter-signal direction differences
        delta_cum_seq_ = [delta_curr]  # doa differences w.r.t first signal
        delta_cum_seq = np.concatenate([[0], delta_cum_seq_])  # the first signal included
        delta_sum = np.sum(delta_curr)  # direction difference between first and last signals
        NUM_STEP = int((doa_max - doa_min - delta_sum) / step)  # number of scanning steps

        for step_idx in range(NUM_STEP):
            doa_first = doa_min + step * step_idx
            DOA = delta_cum_seq + doa_first

            for rep_idx in range(NUM_REPEAT_SS):
                add_noise = np.random.randn(M, N) + 1j * np.random.randn(M, N)
                array_signal = 0
                for ki in range(K):
                    signal_i = 10 ** (SNR[ki] / 20) * (np.random.randn(1, N) + 1j * np.random.randn(1, N))
                    # phase_shift_unit = 2 * np.pi * d / wavelength * np.sin(DOA / 180 * np.pi)
                    array_geom = np.expand_dims(np.array(np.arange(M)), axis=-1) * d
                    phase_shift_array = 2 * np.pi * array_geom / wavelength * np.sin(DOA[ki] / 180 * np.pi)
                    a_i = np.cos(phase_shift_array) + 1j * np.sin(phase_shift_array)
                    array_signal_i = np.matmul(a_i, signal_i)
                    array_signal += array_signal_i

                array_output_nf = array_signal + 0 * add_noise  # noise-free output
                array_output = array_signal + 1 * add_noise

                array_covariance_nf = 1 / N * (np.matmul(array_output_nf, np.matrix.getH(array_output_nf)))
                array_covariance = 1 / N * (np.matmul(array_output, np.matrix.getH(array_output)))
                cov_vector_nf_ = []
                cov_vector_ = []
                for row_idx in range(M):
                    cov_vector_nf_.extend(array_covariance_nf[row_idx, (row_idx + 1):])
                    cov_vector_.extend(array_covariance[row_idx, (row_idx + 1):])
                cov_vector_nf_ = np.asarray(cov_vector_nf_)
                cov_vector_nf_ext = np.concatenate([cov_vector_nf_.real, cov_vector_nf_.imag])
                cov_vector_nf = 1 / np.linalg.norm(cov_vector_nf_ext) * cov_vector_nf_ext
                data_train_ss['input_nf'].append(cov_vector_nf)
                cov_vector_ = np.asarray(cov_vector_)
                cov_vector_ext = np.concatenate([cov_vector_.real, cov_vector_.imag])
                cov_vector = 1 / np.linalg.norm(cov_vector_ext) * cov_vector_ext
                data_train_ss['input'].append(cov_vector)
                # construct spatial spectrum target
                target_spectrum = generate_target_spectrum(DOA, doa_min, grid_ss, NUM_GRID_SS)
                data_train_ss['target_spec'].append(target_spectrum)
    print(np.shape(data_train_ss['input']))
    print(np.shape(data_train_ss['target_spec']))
    return data_train_ss


def generate_training_data_sf_AI(M, N, d, wavelength, SNR, doa_min, NUM_REPEAT_SF, grid, GRID_NUM, output_size,
                                 SF_SCOPE):
    data_train_sf = {}
    data_train_sf['input_nf'] = []
    data_train_sf['input'] = []
    data_train_sf['target_spec'] = []
    for doa_idx in range(GRID_NUM):
        DOA = doa_min + grid * doa_idx

        for rep_idx in range(NUM_REPEAT_SF):
            add_noise = np.random.randn(M, N) + 1j * np.random.randn(M, N)
            array_signal = 0

            signal_i = 10 ** (SNR / 20) * (np.random.randn(1, N) + 1j * np.random.randn(1, N))
            # phase_shift_unit = 2 * np.pi * d / wavelength * np.sin(DOA / 180 * np.pi)
            array_geom = np.expand_dims(np.array(np.arange(M)), axis=-1) * d
            phase_shift_array = 2 * np.pi * array_geom / wavelength * np.sin(DOA / 180 * np.pi)
            a_i = np.cos(phase_shift_array) + 1j * np.sin(phase_shift_array)
            array_signal_i = np.matmul(a_i, signal_i)
            array_signal += array_signal_i

            array_output_nf = array_signal + 0 * add_noise  # noise-free output
            array_output = array_signal + 1 * add_noise

            array_covariance_nf = 1 / N * (np.matmul(array_output_nf, np.matrix.getH(array_output_nf)))
            array_covariance = 1 / N * (np.matmul(array_output, np.matrix.getH(array_output)))
            cov_vector_nf_ = []
            cov_vector_ = []
            for row_idx in range(M):
                cov_vector_nf_.extend(array_covariance_nf[row_idx, (row_idx + 1):])
                cov_vector_.extend(array_covariance[row_idx, (row_idx + 1):])
            cov_vector_nf_ = np.asarray(cov_vector_nf_)
            cov_vector_nf_ext = np.concatenate([cov_vector_nf_.real, cov_vector_nf_.imag])
            cov_vector_nf = 1 / np.linalg.norm(cov_vector_nf_ext) * cov_vector_nf_ext
            data_train_sf['input_nf'].append(cov_vector_nf)
            cov_vector_ = np.asarray(cov_vector_)
            cov_vector_ext = np.concatenate([cov_vector_.real, cov_vector_.imag])
            cov_vector = 1 / np.linalg.norm(cov_vector_ext) * cov_vector_ext
            data_train_sf['input'].append(cov_vector)
            # construct multi-task autoencoder target
            scope_label = int((DOA - doa_min) / SF_SCOPE)
            target_curr_pre = np.zeros([output_size * scope_label, 1])
            target_curr_post = np.zeros([121, 1])
            target_curr = np.expand_dims(cov_vector, axis=-1)
            target = np.concatenate([target_curr_pre, target_curr, target_curr_post], axis=0)
            data_train_sf['target_spec'].append(np.squeeze(target))
    print(np.shape(data_train_sf['input']))
    print(np.shape(data_train_sf['target_spec']))
    np.savetxt('data', data_train_sf['input'])
    np.savetxt('label', data_train_sf['target_spec'])
    return data_train_sf


def generate_array_cov_vector_AI(M, N, d, wavelength, DOA, SNR):
    doa_min = -60
    grid_ss = 1
    NUM_GRID_SS = 121
    data_train_ss = {}
    data_train_ss['input_nf'] = []
    data_train_ss['input'] = []
    data_train_ss['target_spec'] = []
    K = len(DOA)

    add_noise = np.random.randn(M, N) + 1j * np.random.randn(M, N)
    array_signal = 0
    for ki in range(K):
        signal_i = 10 ** (SNR[ki] / 20) * (np.random.randn(1, N) + 1j * np.random.randn(1, N))
        # phase_shift_unit = 2 * np.pi * d / wavelength * np.sin(DOA / 180 * np.pi)
        array_geom = np.expand_dims(np.array(np.arange(M)), axis=-1) * d
        phase_shift_array = 2 * np.pi * array_geom / wavelength * np.sin(DOA[ki] / 180 * np.pi)
        a_i = np.cos(phase_shift_array) + 1j * np.sin(phase_shift_array)
        # a_i = np.matmul(AP_mtx, a_i)
        # a_i = np.matmul(MC_mtx, a_i)
        array_signal_i = np.matmul(a_i, signal_i)
        array_signal += array_signal_i

    array_output = array_signal + add_noise

    array_covariance = 1 / N * (np.matmul(array_output, np.matrix.getH(array_output)))
    cov_vector_ = []
    for row_idx in range(M):
        cov_vector_.extend(array_covariance[row_idx, (row_idx + 1):])
    cov_vector_ = np.asarray(cov_vector_)
    cov_vector_ext = np.concatenate([cov_vector_.real, cov_vector_.imag])
    cov_vector = 1 / np.linalg.norm(cov_vector_ext) * cov_vector_ext
    data_train_ss['input'].append(cov_vector)
    target_spectrum = generate_target_spectrum(DOA, doa_min, grid_ss, NUM_GRID_SS)
    data_train_ss['target_spec'].append(target_spectrum)

    # construct spatial spectrum target
    return cov_vector


def get_DOA_estimate(spec, DOA, doa_min, grid):
    K = len(DOA)

    # extract peaks from spectrum
    peaks = []
    peak_flag = False
    peak_start = 0
    peak_end = 0
    idx = 0
    while idx < len(spec):
        if spec[idx][0] > 0:
            if peak_flag == False:
                peak_start = idx
                peak_end = idx
            else:
                peak_end += 1
            peak_flag = True
        else:
            if peak_flag == True:
                peak_curr = np.array([peak_start, peak_end])
                peaks.append(peak_curr)
            peak_flag = False
        idx += 1

    # estimate directions
    K_est = len(peaks)
    peak_doa_list = []
    peak_amp_list = []
    for ki in range(K_est):
        curr_start = peaks[ki][0]
        curr_end = peaks[ki][1]
        curr_spec = [spec[ii][0] for ii in range(curr_start, curr_end + 1)]
        curr_grid = doa_min + grid * np.arange(curr_start, curr_end + 1)
        curr_amp = np.sum(curr_spec)  # sort peaks with total energy
        curr_doa = np.sum(curr_spec * curr_grid) / np.sum(curr_spec)
        peak_doa_list.append(curr_doa)
        peak_amp_list.append(curr_amp)

    # output doa estimates
    doa_est = []
    if K_est == 0:
        for ki in range(K):
            doa_est.append(DOA[0])
    elif K_est <= K:
        for ki in range(K):
            doa_i = DOA[ki]
            est_error = [np.abs(peak_doa - doa_i) for peak_doa in peak_doa_list]
            est_idx = np.argmin(est_error)
            doa_est_i = peak_doa_list[est_idx]
            doa_est.append(doa_est_i)
    else:
        doa_est_ = []
        for ki in range(K):
            est_idx = np.argmax(peak_amp_list)
            doa_est_i = peak_doa_list[est_idx]
            doa_est_.append(doa_est_i)
            peak_amp_list[est_idx] = -1
        for ki in range(K):
            doa_i = DOA[ki]
            est_error = [np.abs(peak_doa - doa_i) for peak_doa in doa_est_]
            est_idx = np.argmin(est_error)
            doa_est_i = doa_est_[est_idx]
            doa_est.append(doa_est_i)

    return doa_est


import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()
tf.reset_default_graph()


class Ensemble_Model():
    def __init__(self, input_size_ss, _weights, _biases, n_hidden, learning_rate_ss):
        # DoA Prediction
        # place holder
        self.data_train_ = tf.placeholder(tf.float32, shape=[None, input_size_ss])
        # self.label_ss_ = tf.placeholder(tf.float32, shape=[None, output_size_ss * SF_NUM])
        self.label_ss_ = tf.placeholder(tf.float32, shape=[None, 121])
        self.label_ss = tf.transpose(self.label_ss_)
        self.data_train1 = tf.transpose(self.data_train_)  # permute n_steps and batch_size
        # Reshape to prepare input to hidden activation
        self.data_train = tf.reshape(self.data_train1, [-1, input_size_ss])
        self.data_train = tf.nn.relu(tf.matmul(self.data_train_, _weights['hidden']) + _biases['hidden'])
        # Split data because rnn cell needs a list of inputs for the RNN inner loop
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
        self.output_ss = tf.transpose(self.output_ss)
        # loss and optimizer
        self.error_ss = self.label_ss - self.output_ss
        self.loss_ss = tf.reduce_mean(tf.norm(tf.square(self.error_ss), ord=1))
        # Tối ưu hóa loss với adam
        self.train_op_ss = tf.train.AdamOptimizer(
            learning_rate=learning_rate_ss, beta1=0.9, beta2=0.999, epsilon=1e-08, use_locking=False,
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


# Para
# # array signal parameters
fc = 2e9  # Tần số tín hiệu đến
c = 3e8  # Vận tốc ánh sáng
M = 10  # Số phần tử anten
N = 400  # snapshot number
wavelength = c / fc  # Bước sóng
d = 0.5 * wavelength  # Khoảng cách giữa các phần tử anten
K_ss = 2  # signal number
doa_min = -60  # DOA max (degree)
doa_max = 60  # DOA min (degree)
# # spatial filter training parameters

grid_sf = 1  # DOA step (degree) for generating different scenarios
# GRID_NUM_SF = int((doa_max - doa_min) / grid_sf)
GRID_NUM_SF = int((doa_max - doa_min) / grid_sf)
SF_NUM = 6  # number of spatial filters
SF_SCOPE = (doa_max - doa_min) / SF_NUM  # spatial scope of each filter
SNR_sf = 10
NUM_REPEAT_SF = 1  # number of repeated sampling with random noise

noise_flag_sf = 1  # 0: noise-free; 1: noise-present
amp_or_phase = 0  # show filter amplitude or phase: 0-amplitude; 1-phase
NUM_GRID_SS = 121
# # training set parameters
# SS_SCOPE = SF_SCOPE / SF_NUM   # scope of signal directions
step_ss = 1  # DOA step (degree) for generating different scenarios

doa_delta = np.array(np.arange(20) + 1) * 0.1 * SF_SCOPE
# doa_delta = np.array(np.arange(122)) # inter-signal direction differences
SNR_ss = np.array([10, 10, 10]) + 0
NUM_REPEAT_SS = 50  # number of repeated sampling with random noise

noise_flag_ss = 1  # 0: noise-free; 1: noise-present

# # DNN parameters
grid_ss = 1  # inter-grid angle in spatial spectrum

input_size_ss = M * (M - 1)  # 90
batch_size_ss = 32
learning_rate_ss = 0.001
num_epoch_ss = 400
n_hidden = 256
n_classes = 121

weights = {
    'hidden': tf.Variable(tf.random_normal([input_size_ss, n_hidden])),  # Hidden layer weights
    'out': tf.Variable(tf.random_normal([n_hidden, n_classes], mean=1.0))
}
biases = {
    'hidden': tf.Variable(tf.random_normal([n_hidden])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}

# test_DOA = np.array([31, 41])
# test_K = len(test_DOA)
# test_SNR = np.array([10, 10])
num_epoch_test = 1000
RMSE = []
#
# # Create data
data_train_ss = generate_training_data_ss_AI(M, N, K_ss, d, wavelength, SNR_ss, doa_min, doa_max, step_ss, doa_delta,
                                           NUM_REPEAT_SS, grid_ss, NUM_GRID_SS)
save_path = 'new_model2/'
# # Model
enmod_2 = Ensemble_Model(input_size_ss, weights, biases, n_hidden, learning_rate_ss)
#
# Train
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    print('Training...')
    # train
    for epoch in range(num_epoch_ss):
        [data_batches, label_batches] = generate_spec_batches(data_train_ss, batch_size_ss, noise_flag_ss)
        # print(np.shape(data_batches))
        for batch_idx in range(len(data_batches)):
            data_batch = data_batches[batch_idx]
            # print(data_batch)
            label_batch = label_batches[batch_idx]
            feed_dict = {enmod_2.data_train_: data_batch, enmod_2.label_ss_: label_batch}
            _, accuracy, loss_ss = sess.run([enmod_2.train_op_ss, enmod_2.accuracy, enmod_2.loss_ss],
                                            feed_dict=feed_dict)
        print('Epoch : {:03d}, Training: Loss - {:.4f}, Accuracy - {:.4f}%'.format(epoch + 1, loss_ss, accuracy))

    tf.train.Saver()
    saver = tf.train.Saver()
    saver.save(sess, save_path + 'modelLSTM')
#
#     global graph
#     graph = tf.get_default_graph()
#     print('testing...')
#     # saver.restore(sess,save_path + 'modelLSTM')
#     # # test
#     est_DOA = []
#     MSE_rho = np.zeros([test_K, ])
#     for epoch in range(num_epoch_test):
#         with graph.as_default():
#             test_cov_vector = generate_array_cov_vector_AI(M, N, d, wavelength, test_DOA, test_SNR)
#             # data_batch = np.expand_dims(test_cov_vector, axis=-1)
#             # data_batch = data_batch.tolist()
#             #
#             test_cov_vector = test_cov_vector.reshape(1, test_cov_vector.shape[0])
#             data_batch = []
#             data_batch.append(test_cov_vector[0])
#             # print(data_batch.shape)
#             #
#             feed_dict = {enmod_2.data_train_: data_batch}
#             ss_output = sess.run(enmod_2.output_ss, feed_dict=feed_dict)
#             ss_min = np.min(ss_output)
#             ss_output_regularized = [ss if ss > -ss_min else [0.0] for ss in ss_output]
#
#             est_DOA_ii = get_DOA_estimate(ss_output, test_DOA, doa_min, grid_ss)
#             est_DOA.append(est_DOA_ii)
#             MSE_rho += np.square((est_DOA_ii - test_DOA))
#
#     RMSE_rho = np.sqrt(MSE_rho / (num_epoch_test))
#     RMSE.append(RMSE_rho)
#
# print("Act_DoA", test_DOA)
# print("est_DoA", est_DOA)
#
# print("----------")
# print('RMSE: ', RMSE)
# print('RMSE mean: ', np.mean(RMSE))
#
# Test
test_DOA = np.array([10,20])
test_SNR = [-10, -10]
test_K = len(test_DOA)
# test_SNR_total = np.array([[-10, -10], [-8, -8], [-6, -6], [-4, -4],[-2, -2], [0, 0], [2,2], [4,4],[6,6],[8,8],[10,10]])
num_epoch_test = 1000

global graph
graph = tf.get_default_graph()
save_path = 'new_model2/'

RMSE = []
print(doa_delta)
with tf.Session() as sess:
    tf.train.Saver()
    saver = tf.train.Saver()
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    print('testing...')
    saver.restore(sess, save_path + 'modelLSTM')
    # # test
    est_DOA = []

    MSE_rho = np.zeros([test_K, ])

    print(test_SNR)
    for epoch in range(num_epoch_test):
        with graph.as_default():
            test_cov_vector = generate_array_cov_vector_AI(M, N, d, wavelength, test_DOA, test_SNR)
            # data_batch = np.expand_dims(test_cov_vector, axis=-1)
            # data_batch = data_batch.tolist()
            #
            test_cov_vector = test_cov_vector.reshape(1, test_cov_vector.shape[0])
            data_batch = []
            data_batch.append(test_cov_vector[0])
            # print(data_batch.shape)
            #
            feed_dict = {enmod_2.data_train_: data_batch}
            ss_output = sess.run(enmod_2.output_ss, feed_dict=feed_dict)
            ss_min = np.min(ss_output)
            ss_output_regularized = [ss if ss > -ss_min else [0.0] for ss in ss_output]

            est_DOA_ii = get_DOA_estimate(ss_output, test_DOA, doa_min, grid_ss)
            est_DOA.append(est_DOA_ii)
            MSE_rho += np.square((est_DOA_ii - test_DOA))
    print(MSE_rho)
    sum = np.sum(MSE_rho)
    print(sum)

    RMSE_rho = np.sqrt(sum / (num_epoch_test * K_ss))

    print("Act_DoA", test_DOA)
    print("est_DoA", est_DOA)

    print("----------")
    print('RMSE: ', RMSE_rho)
    RMSE.append(RMSE_rho)
print(RMSE)
