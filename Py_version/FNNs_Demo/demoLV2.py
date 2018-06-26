#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/6/15 13:17
# @Author  : zzy824
# @File    : demoLV2.py

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
tf.set_random_seed(55)
np.random.seed(55)

""" add some branched based on demoLV1 """

class FNN(object):
    """Build a general FeedForward neural network
    :param
    ----------
    learning_rate: float
    drop_out: float
    Layers: list
        The number of layers
    N_hidden: list
        The number of nodes in layers
    D_input: int
        Input dimension
    D_label: int
        Label dimension
    Task_type: string
        'regression' or 'classification'
    L2_lambda: float
    First_Author : YJango; 2016/11/25
    Second_Author: zzy824;2018/6/15
    """
    def __init__(self, learning_rate, drop_keep, Layers, N_hidden,
                 D_input, D_label, Task_type='regression', L2_lambda=0.0):
        # the whole sharing attribute
        self.learning_rate = learning_rate
        self.drop_keep = drop_keep
        self.Layers = Layers
        self.N_hidden = N_hidden
        self.D_input = D_input
        self.D_label = D_label
        # loss function controled by Task_type
        self.Task_type = Task_type
        # L2 regularizition's strength
        self.L2_lambda = L2_lambda
        # store L2 regularization for each layer
        self.l2_penalty = tf.constant(0.0)
        # hid_layers for storing output of all hidden layers
        self.hid_layers = []
        # W for storing weights of all layers
        self.W = []
        # b for storing biases of all layers
        self.b = []
        # total_l2 for storing L2 of all layers
        self.total_l2 = []
        # those parameters will be define in "build" function
        self.train_step = None
        self.output = None
        self.loss = None
        self.accuracy = None
        self.total_loss = None

        # for generating figures of tensorflow
        with tf.name_scope('Input'):
            self.inputs = tf.placeholder(tf.float32, [None, D_input], name="inputs")
        with tf.name_scope('Label'):
            self.labels = tf.placeholder(tf.float32, [None, D_label], name='labels')
        with tf.name_scope('keep_rate'):
            self.drop_keep_rate = tf.placeholder(tf.float32, name='dropout_keep')

        # generate when initialize
        self.build('F')

    @staticmethod
    def weight_init(shape):
        """Initialize weight of neural network and initialization could be changed here

        Args:
          shape: list [in_dim, out_dim]

        Returns:
          a Varible which is initialized by random_uniform
        """
        initial = tf.random_uniform(shape, minval=-np.sqrt(5) * np.sqrt(1.0 / shape[0]),
                                    maxval=np.sqrt(5) * np.sqrt(1.0 / shape[0]))
        return tf.Variable(initial)

    @staticmethod
    def bias_init(shape):
        """Initialize weight of neural network and initialization could be changed here

        Args:
          shape: list [in_dim, out_dim]

        Returns:
          a Varible which is initialize by a constant
        """
        # can change initialization here
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)

    @staticmethod
    def variable_summaries(var, name):
        """For recording data in training process

        Args:
            var: numbers for calculating
            name: names for name_scope
        """
        # generate two figures display sum and mean
        with tf.name_scope(name + '_summaries'):
            mean = tf.reduce_mean(var)
            tf.summary.scalar('mean_' + name, mean)
        with tf.name_scope(name + '_stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        # record changes in value after each time training
        tf.summary.scalar('_stddev_' + name, stddev)
        tf.summary.scalar('_max_' + name, tf.reduce_max(var))
        tf.summary.scalar('_min_' + name, tf.reduce_min(var))
        tf.summary.histogram(name=name, values=var)

    def layer(self, in_tensor, in_dim, out_dim, layer_name, act=tf.nn.relu):
        """ a Fuction for establishing each neural layer

        Args:
        :param in_tensor:
        :param in_dim:
        :param out_dim:
        :param layer_name:
        :param act:
        :return:
        """
        with tf.name_scope(layer_name):
            with tf.name_scope(layer_name+'_weights'):
                # initialize weight with weight_init()
                weights = self.weight_init([in_dim, out_dim])
                # self.W will state before usage of this function
                self.W.append(weights)
                # count weight
                self.variable_summaries(weights, layer_name + '_weights')
            with tf.name_scope(layer_name + 'biases'):
                biases = self.bias_init([out_dim])
                # self.b will state before usage of this function
                self.b.append(biases)
                self.variable_summaries(biases, layer_name + '_biases')
            with tf.name_scope(layer_name + '_Wx_plus_b'):
                # calculate Wx+b
                pre_activate = tf.matmul(in_tensor, weights) + biases
                # count histogram
                tf.summary.histogram(layer_name + '_pre_activations', pre_activate)
            # calculate a(Wx+b)
        activations = act(pre_activate, name='activation')
        tf.summary.histogram(layer_name + '_activations', activations)
        # return with output of this layer and L2_loss of weight
        return activations, tf.nn.l2_loss(weights)

    def drop_layer(self, in_tensor):
        """ dropout layer of nerual network

        :param in_tensor:
        :return:
        """
        # tf.scalar_summary('dropout_keep', self.drop_keep_rate)
        dropped = tf.nn.dropout(in_tensor, self.drop_keep_rate)
        return dropped

    def build(self, prefix):
        # build network
        # incoming represent the position of current tensor
        incoming = self.inputs
        # if not hidden layer
        if self.Layers != 0:
            layer_nodes = [self.D_input] + self.N_hidden
        else:
            layer_nodes = [self.D_input]

        # build hidden layers
        for l in range(self.Layers):
            # build layers through self.layers and refresh the position of incoming
            incoming, l2_loss = self.layer(incoming, layer_nodes[l], layer_nodes[l + 1], prefix + '_hid_' + str(l + 1),
                                           act=tf.nn.relu)
            # count l2
            self.total_l2.append(l2_loss)
            # print some messages of what happened in nerual network
            print('Add dense layer: relu with drop_keep:%s' % self.drop_keep)
            print('    %sD --> %sD' % (layer_nodes[l], layer_nodes[l + 1]))
            # store outputs of hidden layer
            self.hid_layers.append(incoming)
            # add dropout layer
            incoming = self.drop_layer(incoming)

        # build output layer as activation functions usually change with specific tasks:
        # if the task is regression then we will use tf.identity rather than activation function
        if self.Task_type == 'regression':
            out_act = tf.identity
        else:
            # if the task is classification then we will use softmax to fitting probability
            out_act = tf.nn.softmax

        self.output, l2_loss = self.layer(incoming, layer_nodes[-1], self.D_label, layer_name='output', act=out_act)
        print('Add output layer: linear')
        print('    %sD --> %sD' % (layer_nodes[-1], self.D_label))

        # l2 loss's zoom figure
        with tf.name_scope('total_l2'):
            for l2 in self.total_l2:
                self.l2_penalty += l2
            tf.summary.scalar('l2_penalty', self.l2_penalty)
        # loss of different figures:
        # if task's type is regression, the loss function is for judging difference value
        # between prediction and actual value
        if self.Task_type == 'regression':
            with tf.name_scope('SSE'):
                self.loss = tf.reduce_mean((self.output - self.labels) ** 2)
                tf.summary.scalar('loss', self.loss)
        else:
            # if task's type is classification, the loss function is cross entrophy
            entropy = tf.nn.softmax_cross_entropy_with_logits(logits=self.output, labels=self.labels)
            with tf.name_scope('cross_entropy'):
                self.loss = tf.reduce_mean(entropy)
                tf.scalar_summary('loss', self.loss)
            with tf.name_scope('accuracy'):
                correct_prediction = tf.equal(tf.argmax(self.output, 1), tf.argmax(self.labels, 1))
                self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
                tf.scalar_summary('accuracy', self.accuracy)

        # aggregate all losses
        with tf.name_scope('total_loss'):
            self.total_loss = self.loss + self.l2_penalty * self.L2_lambda
            tf.summary.scalar('total_loss', self.total_loss)

        # operation of training
        with tf.name_scope('train'):
            self.train_step = tf.train.AdamOptimizer(self.learning_rate).minimize(self.total_loss)

    # shuffle function
    @staticmethod
    def shufflelists(lists):
        ri = np.random.permutation(len(lists[1]))
        out = []
        for l in lists:
            out.append(l[ri])
        return out

# prepare some data for training "XOR"
inputs = [[0, 0], [0, 1], [1, 0], [1, 1]]
outputs = [0, 1, 1, 0]
X = np.array(inputs).reshape((4, 1, 2)).astype('int16')
Y = np.array(outputs).reshape((4, 1, 1)).astype('int16')

# generate instance of neural network
ff = FNN(learning_rate=1e-3,
         drop_keep=1.0,
         Layers=1,
         N_hidden=[2],
         D_input=2,
         D_label=1,
         Task_type='regression',
         L2_lambda=1e-2)

# loading
sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())
merged = tf.summary.merge_all()
train_writer = tf.summary.FileWriter('log' + '/train', sess.graph)
# print weights before training
W0 = sess.run(ff.W[0])
W1 = sess.run(ff.W[1])
print('W_0:\n%s' % sess.run(ff.W[0]))
print('W_1:\n%s' % sess.run(ff.W[1]))

plt.scatter([1, 1, 5], [1, 3, 2], color=['red', 'red', 'blue'], s=200, alpha=0.4, marker='o')
plt.scatter([3, 3], [1, 3], color=['green', 'green'], s=200, alpha=0.4, marker='o')

plt.plot([1, 3], [1, 1], color='orange', linewidth=abs(W0[0, 0]))
plt.annotate('%0.2f' % W0[0, 0], xy=(2, 1.0))

plt.plot([1, 3], [3, 1], color='blue', linewidth=abs(W0[1, 0]))
plt.annotate('%0.2f' % W0[1, 0], xy=(1.5, 1.5))

plt.plot([1, 3], [1, 3], color='blue', linewidth=abs(W0[0, 1]))
plt.annotate('%0.2f' % W0[0, 1], xy=(1.5, 2.5))

plt.plot([1, 3], [3, 3], color='orange', linewidth=abs(W0[1, 1]))
plt.annotate('%0.2f' % W0[1, 1], xy=(2, 3))

plt.plot([3, 5], [1, 2], color='blue', linewidth=abs(W1[0]))
plt.annotate('%0.2f' % W1[0], xy=(4, 1.5))

plt.plot([3, 5], [3, 2], color='blue', linewidth=abs(W1[1]))
plt.annotate('%0.2f' % W1[1], xy=(4, 2.5))

# output before training
pY = sess.run(ff.output, feed_dict={ff.inputs: X.reshape((4, 2)), ff.drop_keep_rate: 1.0})
print(pY)
plt.scatter([0, 1, 2, 3], pY, color=['red', 'green', 'blue', 'black'], s=25, alpha=0.4, marker='o')
# hidden layer's output before training
pY = sess.run(ff.hid_layers[0], feed_dict={ff.inputs:X.reshape((4,2)),ff.drop_keep_rate:1.0})
print(pY)
plt.scatter(pY[:, 0], color=['red', 'green', 'blue', 'black'], s=25, alpha=0.4, marker='o')
# training section and record
k = 0.0
for i in range(10000):
    k += 1
    summary, _ = sess.run([merged, ff.train_step], feed_dict={ff.inputs: X.reshape((4,2)),ff.labels: Y.reshape((4, 1)), ff.drop_keep_rate: 1.0})
    train_writer.add_summary(summary, k)

# weights after training
W0 = sess.run(ff.W[0])
W1 = sess.run(ff.W[1])
print('W_0:\n%s' % sess.run(ff.W[0]))
print('W_1:\n%s' % sess.run(ff.W[1]))

plt.scatter([1, 1, 5],[1, 3, 2], color=['red', 'red', 'blue'], s=200, alpha=0.4, marker='o')
plt.scatter([3, 3], [1, 3], color=['green', 'green'], s=200, alpha=0.4, marker='o')

plt.plot([1, 3], [1, 1], color='orange', linewidth=abs(W0[0, 0]))
plt.annotate('%0.2f' % W0[0, 0], xy=(2, 1.0))

plt.plot([1, 3], [3, 1], color='blue', linewidth=abs(W0[1, 0]))
plt.annotate('%0.2f' % W0[1, 0], xy=(1.5, 1.5))

plt.plot([1, 3], [1, 3], color='blue', linewidth=abs(W0[0, 1]))
plt.annotate('%0.2f' % W0[0, 1], xy=(1.5, 2.5))

plt.plot([1, 3], [3, 3], color='orange', linewidth=abs(W0[1, 1]))
plt.annotate('%0.2f' % W0[1, 1], xy=(2, 3))

plt.plot([3, 5], [1, 2], color='blue', linewidth=abs(W1[0]))
plt.annotate('%0.2f' % W1[0], xy=(4, 1.5))

plt.plot([3, 5], [3, 2],color='blue', linewidth=abs(W1[1]))
plt.annotate('%0.2f' % W1[1], xy=(4, 2.5))

# output after training
pY = sess.run(ff.output, feed_dict={ff.inputs: X.reshape((4, 2)), ff.drop_keep_rate: 1.0})
print(pY)
plt.scatter([0, 1, 2, 3], pY, color=['red', 'green', 'blue', 'black'], s=25, alpha=0.4, marker='o')

