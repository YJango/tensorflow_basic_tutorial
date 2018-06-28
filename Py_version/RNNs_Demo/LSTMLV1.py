#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/6/19 14:57
# @Author  : zzy824
# @File    : LSTMLV1.py

import tensorflow as tf
import numpy as np
import time
import matplotlib.pyplot as plt

""" functions for data processing """
def Standardize(seq):
    # substract mean
    centerized = seq - np.mean(seq, axis=0)
    # divide standard deviation
    normalized = centerized / np.std(centerized, axis=0)
    return normalized
# read input and output data
mfc = np.load('X.npy')
art = np.load('Y.npy')
totalsamples = len(mfc)
# 20% of total data as validation set
vali_size = 0.2

def data_prer(X, Y):
    """aggregate input and output data into list then concrete all the samples into a new list

    :param X: [n_samples, n_steps, D_input]
    :param Y: [n_samples, D_output]
    :return: list
    """
    D_input = X[0].shape[1]
    pre_data = []
    for x, y in zip(X, Y):
        pre_data.append([Standardize(x).reshape((1, -1, D_input)).astype("float32"),
                         Standardize(y).astype("float32")])
    return pre_data

# processing data
data = data_prer(mfc, art)
# divide data into training set and validation set
train = data[int(totalsamples*vali_size):]
test = data[:int(totalsamples*vali_size)]

print('num of train sequences:%s' % len(train))
print('num of test sequences:%s' % len(test))
print('shape of inputs:', test[0][0].shape)
print('shape of labels:', test[0][1].shape)

""" initialize weights"""
def weight_init(shape):
    initial = tf.random_uniform(shape, minval=-np.sqrt(1.0 / shape[0]), maxval=np.sqrt(5) * np.sqrt(1.0 / shape[0]))
    return tf.Variable(initial, trainable=True)
# initialize them to 0
def zero_init(shape):
    initial = tf.Variable(tf.zeros(shape))
    return tf.Variable(initial, trainable=True)
# initialize othogonal matrix
def orthogonal_initializer(shape, scale = 1.0):
    scale = 1.0
    flat_shape = (shape[0], np.prod(shape[1:]))
    a = np.random.normal(0.0, 1.0, flat_shape)
    u, _, v = np.linalg.svd(a, full_matrices=False)
    q = u if u.shape == flat_shape else v
    q = q.reshape(shape)  # this needs to be corrected to float32
    return tf.Variable(scale * q[:shape[1]], trainable=True, dtype=tf.float32)
# initialize biases
def bias_init(shape):
    initial = tf.constant(0.01, shape=shape)
    return tf.Variable(initial)
# shuffle function
def shufflelists(unsfled_data):
    ri = np.random.permutation(len(unsfled_data))
    sfled_data = [unsfled_data[i] for i in ri]
    return sfled_data

""" class for LSTM model """
class LSTMcell(object):
    def __init__(self, incoming, D_input, D_cell, initializer, f_bias=1.0):
        # vars of LSTM
        # incoming is a data structure for receive data. [n_sample, n_steps, D_cell]
        self.incoming = incoming
        # dimension of input
        self.D_input = D_input
        # dimension of LSTM's hidden state meanwhile of memory cell
        self.D_cell = D_cell
        # parameters of LSTM
        # three parameters of input_gate: igate = W_xi.* x + W_hi.* h + b_i
        self.W_xi = initializer([self.D_input, self.D_cell])
        self.W_hi = initializer([self.D_cell, self.D_cell])
        self.b_i = tf.Variable(tf.zeros([self.D_cell]))
        # three parameters of forget_gate: fgate = W_xf.* x + W_hf.* h + b_f
        self.W_xf = initializer([self.D_input, self.D_cell])
        self.W_hf = initializer([self.D_cell, self.D_cell])
        self.b_f = tf.Variable(tf.constant(f_bias, shape=[self.D_cell]))
        # three parameters of output_gate:ogate = W_xo.* x + W_ho.* h + b_o
        self.W_xo = initializer([self.D_input, self.D_cell])
        self.W_ho = initializer([self.D_cell, self.D_cell])
        self.b_o = tf.Variable(tf.zeros([self.D_cell]))
        # three parameters of calculating new messages
        # cell = W_xc.* x + W_hc.* h + b_c
        self.W_xc = initializer([self.D_input, self.D_cell])
        self.W_hc = initializer([self.D_cell, self.D_cell])
        self.b_c = tf.Variable(tf.zeros([self.D_cell]))
        # set values of hidden state and memory cell whose shape is [n_sample, D_cell]
        init_for_both = tf.matmul(self.incoming[:, 0, :], tf.zeros([self.D_input, self.D_cell]))
        self.hid_init = init_for_both
        self.cell_init = init_for_both
        # combine hideen state and memory cell
        self.previous_h_c_tuple = tf.stack([self.hid_init, self.cell_init])
        # transform data's shape([n_samples, n_steps, D_cell]) into shape [n_steps, n_samples, D_cell]
        self.incoming = tf.transpose(self.incoming, perm=[1, 0, 2])

    def one_step(self, previous_h_c_tuple, current_x):
        # to split hidden state and cell
        prev_h, prev_c = tf.unstack(previous_h_c_tuple)

        # computing
        # input gate
        i = tf.sigmoid(
            tf.matmul(current_x, self.W_xi) +
            tf.matmul(prev_h, self.W_hi) +
            self.b_i)
        # forget Gate
        f = tf.sigmoid(
            tf.matmul(current_x, self.W_xf) +
            tf.matmul(prev_h, self.W_hf) +
            self.b_f)
        # output Gate
        o = tf.sigmoid(
            tf.matmul(current_x, self.W_xo) +
            tf.matmul(prev_h, self.W_ho) +
            self.b_o)
        # new cell info
        c = tf.tanh(
            tf.matmul(current_x, self.W_xc) +
            tf.matmul(prev_h, self.W_hc) +
            self.b_c)
        # current cell
        current_c = f * prev_c + i * c
        # current hidden state
        current_h = o * tf.tanh(current_c)

        return tf.stack([current_h, current_c])

    def all_steps(self):
        # inputs shape : [n_sample, n_steps, D_input]
        # outputs shape : [n_steps, n_sample, D_output]
        hstates = tf.scan(fn=self.one_step,
                          elems=self.incoming,
                          initializer=self.previous_h_c_tuple,
                          name='hstates')[:, 0, :, :]
        return hstates

""" build neural network """
D_input = 39
D_label = 24
learning_rate = 7e-5
num_units = 1024
# inputs and labels of samples
inputs = tf.placeholder(tf.float32, [None, None, D_input], name="inputs")
labels = tf.placeholder(tf.float32, [None, D_label], name="labels")
# instantiate LSTM
rnn_cell = LSTMcell(inputs, D_input, num_units, orthogonal_initializer)
# calling scan to calculate all hidden states
rnn0 = rnn_cell.all_steps()
# reshape for output layer
rnn = tf.reshape(rnn0, [-1, num_units])
# output layer
W = weight_init([num_units, D_label])
b = bias_init([D_label])
output = tf.matmul(rnn, W) + b

loss = tf.reduce_mean((output-labels)**2)

train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)

""" session section """
sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())
# training and recording
def train_epoch(EPOCH):
    for k in range(EPOCH):
        train0 = shufflelists(train)
        for i in range(len(train)):
            sess.run(train_step, feed_dict={inputs: train0[i][0], labels: train0[i][1]})
        tl = 0
        dl = 0
        for i in range(len(test)):
            dl += sess.run(loss, feed_dict={inputs: test[i][0], labels: test[i][1]})
        for i in range(len(train)):
            tl += sess.run(loss, feed_dict={inputs: train[i][0], labels: train[i][1]})
        print(k, 'train:', round(tl/83, 3), 'test:', round(dl/20, 3))

t0 = time.time()
train_epoch(10)
t1 = time.time()
print(" %f seconds" % round((t1 - t0), 2))


pY = sess.run(output, feed_dict={inputs: test[10][0]})
plt.plot(pY[:, 8])
plt.plot(test[10][1][:, 8])
plt.title('test')
plt.legend(['predicted', 'real'])


pY = sess.run(output,feed_dict={inputs: train[1][0]})
plt.plot(pY[:, 6])
plt.plot(train[1][1][:, 6])
plt.title('train')
plt.legend(['predicted', 'real'])























