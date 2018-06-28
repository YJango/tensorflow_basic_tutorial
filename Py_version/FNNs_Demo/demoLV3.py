#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/6/19 11:10
# @Author  : zzy824
# @File    : demoLV3.py

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
tf.set_random_seed(55)
np.random.seed(55)

""" add some branched based on demoLV2，X.npy and Y.npy are files including data for demo"""

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
    def __init__(self, learning_rate, Layers, N_hidden,
                 D_input, D_label, Task_type='regression', L2_lambda=0.0):
        # the whole sharing attribute
        self.learning_rate = learning_rate
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
            print('Add dense layer: relu')
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

def Standardize(seq):
    """

    :param seq:
    :return:
    """
    # subtract mean
    centerized = seq-np.mean(seq, axis=0)
    # divide standard deviation
    normalized = centerized/np.std(centerized, axis=0)
    return normalized
def Makewindows(indata, window_size=41):
    outdata = []
    mid = int(window_size/2)
    indata = np.vstack((np.zeros((mid, indata.shape[1])), indata, np.zeros((mid, indata.shape[1]))))
    for index in range(indata.shape[0]-window_size+1):
        outdata.append(np.hstack(indata[index: index + window_size]))
    return np.array(outdata)

# prepare some data for training "XOR"
mfc = np.load('X.npy')
art = np.load('Y.npy')
x = []
y = []
for i in range(len(mfc)):
    x.append(Makewindows(Standardize(mfc[i])))
    y.append(Standardize(art[i]))
vali_size = 20
totalsamples = len(np.vstack(x))
X_train = np.vstack(x)[int(totalsamples/vali_size):].astype("float32")
Y_train = np.vstack(y)[int(totalsamples/vali_size):].astype("float32")

X_test = np.vstack(x)[:int(totalsamples/vali_size)].astype("float32")
Y_test = np.vstack(y)[:int(totalsamples/vali_size)].astype("float32")
# print the shape of train and test data
print(X_train.shape, Y_train.shape, X_test.shape, Y_test.shape)
# generate instance of neural network
ff = FNN(learning_rate=7e-5,
         Layers=5,
         N_hidden=[2048, 1024, 512, 256, 128],
         D_input=1599,
         D_label=24,
         L2_lambda=1e-4)

# loading
sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())
merged = tf.summary.merge_all()
train_writer = tf.summary.FileWriter('log3' + '/train', sess.graph)
test_writer = tf.summary.FileWriter('log3' + '/test')
#
def plots(T, P, i, n=21, length=400):
    m = 0
    plt.figure(figsize=(20, 16))
    plt.subplot(411)
    plt.plot(T[m:m + length, 7], '--')
    plt.plot(P[m:m + length, 7])

    plt.subplot(412)
    plt.plot(T[m:m + length, 8], '--')
    plt.plot(P[m:m + length, 8])

    plt.subplot(413)
    plt.plot(T[m:m + length, 15], '--')
    plt.plot(P[m:m + length, 15])

    plt.subplot(414)
    plt.plot(T[m:m + length, 16], '--')
    plt.plot(P[m:m + length, 16])
    plt.legend(['True', 'Predicted'])
    plt.savefig('epoch' + str(i) + '.png')
    plt.close()
# training and record
k = 0
Batch = 32
for i in range(1):
    idx = 0
    X0, Y0 = ff.shufflelists([X_train, Y_train])
    while idx < X_train.shape[0]:
        summary, _ = sess.run([merged, ff.train_step], feed_dict={ff.inputs: X0[idx:idx+Batch], ff.labels: Y0[idx:idx+Batch], ff.drop_keep_rate: 1.0})  # when set "keep rate = 1" means unuse of dropout
        idx += Batch
        k += 1
        train_writer.add_summary(summary, k)
    # test
    summary, pY, pL = sess.run([merged, ff.output, ff.loss], feed_dict={ff.inputs: X_test, ff.labels: Y_test, ff.drop_keep_rate: 1.0})
    plots(Y_test, pY, i)
    test_writer.add_summary(summary, k)
    print('epoch%s | train_loss:%s |test_loss:%s' % (i, sess.run(ff.loss,feed_dict={ff.inputs: X0, ff.labels: Y0, ff.drop_keep_rate: 1.0}), sess.run(ff.loss, feed_dict={ff.inputs: X_test, ff.labels: Y_test, ff.drop_keep_rate: 1.0})))