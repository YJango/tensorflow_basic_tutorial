#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/6/14 11:40
# @Author  : zzy824
# @File    : demoLV1.py
""" a XOR neural network by TensorFlow"""
import tensorflow as tf
import numpy as np

""" structure of neural network:
    2 dimension input node
    2 hidden node
    1 ouput node
"""

# define parameter
D_input = 2
D_hidden = 2
D_label = 1
lr = 0.0001


""" processing of forward """
# placeholder initialize by [precision, shape of matrix(None means shape can change randomly), name]
x = tf.placeholder(tf.float32, [None, D_input], name="x")
t = tf.placeholder(tf.float32, [None, D_label], name="t")

# initialize W
W_h1 = tf.Variable(tf.truncated_normal([D_input, D_hidden], stddev=0.1), name="W_h")
# initialise b
b_h1 = tf.Variable(tf.constant(0.1, shape=[D_hidden]), name="b_h")
# calculate Wx+b
pre_act_h1 = tf.matmul(x, W_h1) + b_h1
# calculate a(Wx+b)
act_h1 = tf.nn.relu(pre_act_h1, name='act_h')
# initialize output layer
W_o = tf.Variable(tf.truncated_normal([D_hidden, D_label], stddev=0.1), name="W_o")
b_o = tf.Variable(tf.constant(0.1, shape=[D_label]), name="b_o")
pre_act_o = tf.matmul(act_h1, W_o) + b_o
y = tf.nn.relu(pre_act_o, name="act_y")


""" processing of backword """
# define loss function:
loss = tf.reduce_mean((y-t)**2)
# gradient descent: arg: learning rate; optimizer
train_step = tf.train.AdamOptimizer(lr).minimize(loss)

""" prepare data for training """
X = [[0, 0], [0, 1], [1, 0], [1, 1]]
Y = [[0], [1], [1], [0]]
# when using tensorflow the traning data with np.array format is necessary!
# datatype must below 32bit, it is suggest to use ".astype('float32')"
X = np.array(X).astype('int16')
Y = np.array(Y).astype('int16')

""" load neural network """
# the defect of tf.Session() is unobviously,which is that method can not use tenor.eval() .etc
sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())
# training network
""" GD mode: the whole data as input to network. Updating network through the mean
    value of gradient.Code below as annotation """
# T = 30000
# for i in range(T):
#     sess.run(train_step, feed_dict={x: X, t: Y})
""" SGD mode:only one sample as input to network.The advantage is easy 
    to get rid of saddle point and disadvantage is the unsteady updating direction.
    Code below as annotation """
# T = 30000
# for i in range(T):
#     for j in range(X.shape[0]):  # X.shape[0] means the numbers of sample
#         sess.run(train_step, feed_dict={x: [X[j]], t: [Y[j]]})
""" batch-GD:each time calculate gradient of mean value of batch size of data
    as usual:we definate the sample number under 10 as mini-batch-GD.
    Code below as annotation """
# T = 30000
# b_size = 2  # parameter batch_size
# for i in range(T):
#     b_idx = 0  # todo: batch counter, new form of counter!
#     while b_idx < X.shape[0]:
#         sess.run(train_step, feed_dict={x: X[b_idx: b_idx + b_size],
#                                         t: Y[b_idx: b_idx + b_size]})
#         b_idx += b_size  # refresh index of batch

""" shuffle mode: a trick disorganizes the order of data to improve the effect of training """
def shufflelists(lists):
    """function for disorganizing the order of data

    :param lists: [[[0, 0], [0, 1], [1, 0], [1, 1]],[[0], [1], [1], [0]]]
    :return: obvious shuffled list
    """
    ri = np.random.permutation(len(lists[1]))
    out = []
    for l in lists:
        out.append(l[ri])
    return out
T = 30000
b_size = 2
for i in range(T):
    b_idx = 0
    X, Y = shufflelists([X, Y])
    while b_idx < X.shape[0]:
        sess.run(train_step, feed_dict={x: X[b_idx: b_idx + b_size],
                                        t: Y[b_idx: b_idx + b_size]})
        b_idx += b_size
# check out prediction
print sess.run(y, feed_dict={x: X})

# check out hidden layer
print sess.run(act_h1, feed_dict={x: X})

# and any value could be examine by sess.run(*arg)
print sess.run([W_h1, W_o])



