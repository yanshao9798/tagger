# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import math

class HiddenLayer(object):
    """
    Hidden layer with or without bias.
    Input: tensor of dimension (dims*, input_dim)
    Output: tensor of dimension (dims*, output_dim)
    """
    def __init__(self, input_dim, output_dim, bias=True, activation='tanh', name='hidden_layer'):
        """
        :param input_dim:
        :param output_dim:
        :param bias:
        :param activation:
        :param name:
        :return:
        """

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.is_bias = bias
        self.name = name
        if activation == 'linear':
            self.activation = None
        elif activation == 'tanh':
            self.activation = tf.tanh
        elif activation == 'sigmoid':
            self.activation = tf.sigmoid
        elif activation == 'softmax':
            self.activation = tf.nn.softmax
        elif activation is not None:
            raise Exception('Unknown activation function: ' % activation)

        #Initialise weights and bias
        rand_uniform_init = tf.contrib.layers.xavier_initializer()
        self.weights = tf.get_variable(name + '_weights', [input_dim, output_dim], initializer=rand_uniform_init)
        self.bias = tf.get_variable(name + '_bias', [output_dim], initializer=tf.constant_initializer(0.0))

        #define parameters
        if self.is_bias:
            self.params = [self.weights, self.bias]
        else:
            self.params = [self.weights]

    def __call__(self, input_t):
        """
        :param input_t:
        :return:
        """
        self.input = input_t
        self.linear = tf.matmul(self.input, self.weights)
        if self.is_bias:
            self.linear += self.bias
        if self.activation == None:
            self.output = self.linear
        else:
            self.output = self.activation(self.linear)
        return self.output


class EmbeddingLayer(object):
    """
    Embedding layer to map input into word representations
    Input: tensor of dimension (dim*) with values in range(0, input_dim)
    Output: tensor of dimension (dim*, output_dim)
    """
    def __init__(self, input_dim, output_dim, weights=None, is_variable=False, trainable=True, name='embedding_layer'):
        """
        :param input_dim:
        :param output_dim:
        :param name:
        """
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.name = name
        self.trainable = trainable
        self.weights = weights

        # Generate random embeddings or read pre-trained embeddings
        rand_uniform_init = tf.contrib.layers.xavier_initializer()
        if self.weights is None:
            self.embeddings = tf.get_variable(self.name + '_emb', [self.input_dim, self.output_dim], initializer=rand_uniform_init, trainable=self.trainable)
        elif is_variable:
            self.embeddings = weights
        else:
            emb_count = len(weights)
            if emb_count < input_dim:
                padd_weights = np.zeros([self.input_dim - emb_count, self.output_dim], dtype='float32')
                self.weights = np.concatenate((self.weights, padd_weights), axis=0)
            self.embeddings = tf.get_variable(self.name + '_emb', initializer=self.weights, trainable=self.trainable)
        #Define Parameters
        self.params = [self.embeddings]
        self.weight_name = self.name + '_emb'

    def __call__(self, input_t):
        """
        return the embeddings of the given indexes
        :param input:
        :return:
        """
        self.input = input_t
        self.output = tf.gather(self.embeddings, self.input)
        return self.output


class Convolution(object):
    '''
    Regular convolutional layer
    '''
    @staticmethod
    def weight_variable(shape, name):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial, name=name)

    @staticmethod
    def bias_variable(shape, name):
        initial = tf.constant(0.1, shape=shape, name=name)
        return tf.Variable(initial)

    def __init__(self, conv_width, in_channels, out_channels, stride=1, dim=2, padding='SAME', name='convolutional_layer'):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.dim = dim
        if dim == 1:
            self.strides = [1, stride, 1]
        else:
            self.strides = [1, stride, stride, 1]
        self.padding = padding
        self.name = name
        self.conv_width = conv_width
        if dim == 1:
            self.w_conv = self.weight_variable([self.conv_width, self.in_channels, self.out_channels], name=self.name + '_w')
        else:
            self.w_conv = self.weight_variable([self.conv_width, self.conv_width, self.in_channels, self.out_channels], name=self.name + '_w')
        self.b_conv = self.bias_variable([self.out_channels], name=self.name + '_b')

    def conv2d(self, x, W):
        return tf.nn.conv2d(x, W, strides=self.strides, padding=self.padding)

    def conv1d(self, x, W):
        return tf.nn.conv1d(x, W, stride=self.strides, padding=self.padding)

    def __call__(self, input_t):
        if self.dim == 1:
            return tf.nn.relu(self.conv1d(input_t, self.w_conv) + self.b_conv)
        else:
            return tf.nn.relu(self.conv2d(input_t, self.w_conv) + self.b_conv)



class Maxpooling(object):
    '''
    Maxpooling layer
    '''
    def __init__(self, pooling_size, stride=1, padding='SAME', name='pooling_layer'):
        self.padding = padding
        self.name = name
        self.ksize = [1, pooling_size, pooling_size, 1]

    def __call__(self, input_v):
        return tf.nn.max_pool(input_v, ksize=self.ksize, strides=self.ksize, padding='SAME')


class DropoutLayer(object):
    """
    Dropout layer
    """
    def __init__(self, p=0.5, name='dropout_layer'):
        """
        :param p: dropout rate
        :param name:
        """
        #assert 0. <= p < 1
        self.p = p
        self.name = name

    def __call__(self, input_t):
        self.input = input_t
        return tf.nn.dropout(self.input, keep_prob=1 - self.p, name=self.name)


class BiLSTM(object):
    """
    Bidirectional LSTM
    """
    def __init__(self, cell_dim, nums_layers=1, p=0.5, fw_cell=None, bw_cell=None, state=False, name='biLSTM', scope=None):
        """
        :param cell_dim:
        :param nums_steps:
        :param nums_layers:
        :param p:
        :param name:
        """
        self.cell_dim = cell_dim
        self.nums_layers = nums_layers
        self.p = p
        self.state = state
        self.name = name
        self.scope = scope
        if fw_cell is None:
            self.lstm_cell_fw = tf.nn.rnn_cell.LSTMCell(self.cell_dim, state_is_tuple=True)
        else:
            self.lstm_cell_fw = fw_cell
        if bw_cell is None:
            self.lstm_cell_bw = tf.nn.rnn_cell.LSTMCell(self.cell_dim, state_is_tuple=True)
        else:
            self.lstm_cell_bw = bw_cell
        #assert 0. <= p < 1

    def __call__(self, input_t, input_ids):
        self.input = input_t
        self.input_ids = input_ids
        #if self.p > 0.:
        self.lstm_cell_fw = tf.nn.rnn_cell.DropoutWrapper(self.lstm_cell_fw, output_keep_prob=(1 - self.p))
        self.lstm_cell_bw = tf.nn.rnn_cell.DropoutWrapper(self.lstm_cell_bw, output_keep_prob=(1 - self.p))
        if self.nums_layers > 1:
            self.lstm_cell_fw = tf.nn.rnn_cell.MultiRNNCell([self.lstm_cell_fw] * self.nums_layers)
            self.lstm_cell_bw = tf.nn.rnn_cell.MultiRNNCell([self.lstm_cell_bw] * self.nums_layers)
        self.length = tf.reduce_sum(tf.sign(self.input_ids), axis=1)
        self.length = tf.cast(self.length, dtype=tf.int32)
        int_states, final_states = tf.nn.bidirectional_dynamic_rnn(self.lstm_cell_fw, self.lstm_cell_bw, self.input,
                                                                   sequence_length=self.length, dtype=tf.float32,
                                                                   scope=self.scope)
        self.output = tf.concat(values=int_states, axis=2)
        if self.state:
            return self.output, final_states
        else:
            return self.output



class TimeDistributed(object):
    """
    Time-distributed wrapper for layers
    """
    def __init__(self, layer, name='Time-distributed Wrapper'):
        self.layer = layer
        self.name = name

    def __call__(self, input_t, input_ids=None, pad=None):
        self.input = tf.unstack(input_t, axis=1)
        if input_ids is None:
            self.out = [self.layer(splits) for splits in self.input]
        else:
            self.out = []
            pad = self.layer(self.input[0])*0
            masks = tf.reduce_sum(input_ids, axis=0)
            length = len(self.input)
            for i in range(length):
                r = tf.cond(tf.greater(masks[i], 0), lambda: self.layer(input_t[i]), lambda: pad)
                self.out.append(r)
        self.out = tf.stack(self.out, axis=1)
        return self.out


class Forward(object):
    """
    forward algorithm for the CRF loss
    """
    def __init__(self, observations, transitions, nums_tags, length, batch_size, viterbi=True):
        self.observations = observations
        self.transitions = transitions
        self.viterbi = viterbi
        self.length = length
        self.batch_size = batch_size
        self.nums_tags = nums_tags
        self.nums_steps = observations.get_shape().as_list()[1]

    @staticmethod
    def log_sum_exp(x, axis=None):
        """
        Sum probabilities in the log-space
        :param x:
        :param axis:
        :return:
        """
        x_max = tf.reduce_max(x, axis=axis, keep_dims=True)
        x_max_ = tf.reduce_max(x, axis=axis)
        return x_max_ + tf.log(tf.reduce_sum(tf.exp(x - x_max), axis=axis))

    def __call__(self):
        small = -1000
        class_pad = tf.stack(small * tf.ones([self.batch_size, self.nums_steps, 1]))
        self.observations = tf.concat(axis=2, values=[self.observations, class_pad])
        b_vec = tf.cast(tf.stack(([small] * self.nums_tags + [0]) * self.batch_size), tf.float32)
        b_vec = tf.reshape(b_vec, [self.batch_size, 1, -1])
        self.observations = tf.concat(axis=1, values=[b_vec, self.observations])
        self.transitions = tf.reshape(tf.tile(self.transitions, [self.batch_size, 1]), [self.batch_size, self.nums_tags + 1, self.nums_tags + 1])
        self.observations = tf.reshape(self.observations, [-1, self.nums_steps + 1, self.nums_tags + 1, 1])
        self.observations = tf.transpose(self.observations, [1, 0, 2, 3])
        previous = self.observations[0, :, :, :]
        max_scores = []
        max_scores_pre = []
        alphas = [previous]
        for t in range(1, self.nums_steps + 1):
            previous = tf.reshape(previous, [-1, self.nums_tags + 1, 1])
            current =  tf.reshape(self.observations[t,:, :, :], [-1, 1, self.nums_tags + 1])
            alpha_t = previous + current + self.transitions
            if self.viterbi:
                max_scores.append(tf.reduce_max(alpha_t, axis=1))
                max_scores_pre.append(tf.argmax(alpha_t, axis=1))
            alpha_t = tf.reshape(self.log_sum_exp(alpha_t, axis=1), [-1, self.nums_tags + 1, 1])
            alphas.append(alpha_t)
            previous = alpha_t
        alphas = tf.stack(alphas, axis=1)
        alphas = tf.reshape(alphas, [-1, self.nums_tags + 1, 1])
        last_alphas = tf.gather(alphas, tf.range(0, self.batch_size) * (self.nums_steps + 1) + self.length)
        last_alphas = tf.reshape(last_alphas, [self.batch_size, self.nums_tags + 1, 1])
        max_scores = tf.stack(max_scores, axis=1)
        max_scores_pre = tf.stack(max_scores_pre, axis=1)
        return tf.reduce_sum(self.log_sum_exp(last_alphas, axis=1)), max_scores, max_scores_pre


