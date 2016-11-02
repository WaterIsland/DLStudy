#!/usr/local/bin/python
# -*- coding: utf-8 -*-

import tensorflow as tf

# init values
'''
input_size  = 784
hidden_size = 200
output_size = 10
init_mean   = 0.0
init_dev    = 0.1
eta         = 0.01
'''

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')


class ExtendedTensorflowCNN():
    
#    def __init__(self):
        
    def inference(self, input_placeholder):
        # first convolutional layer
        with tf.name_scope('first_convolutional_layer') as scope:
            W_conv1 = weight_variable([5, 5, 1, 32])
            b_conv1 = bias_variable([32])
            h_conv1 = tf.nn.relu(conv2d(input_placeholder, W_conv1) + b_conv1)
            h_pool1 = max_pool_2x2(h_conv1)

        # second convolutional layer
        with tf.name_scope('second_convolutional_layer') as scope:
            W_conv2 = weight_variable([5, 5, 32, 64])
            b_conv2 = bias_variable([64])
            h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
            h_pool2 = max_pool_2x2(h_conv2)

        # Densely Connected Layer
        with tf.name_scope('Densely_Connected_layer') as scope:
            W_fc1 = weight_variable([7 * 7 * 64, 1024])
            b_fc1 = bias_variable([1024])
            h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
            h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

        # Dropout
        with tf.name_scope('Dropout') as scope:
            # probability of drop out.
            # on training, keep_prob = 0.5 conventionally.
            # on testing,  keep_prob = 1.0 conventionally.
            self.keep_prob = tf.placeholder(tf.float32)
            h_fc1_drop = tf.nn.dropout(h_fc1, self.keep_prob)

        # Readout Layer
        with tf.name_scope('Readout_Layer') as scope:
            W_fc2 = weight_variable([1024, 10])
            b_fc2 = bias_variable([10])
            self.y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
       
        return self.y_conv


    def loss(self, logits, output_placeholder):
        with tf.name_scope('loss') as scope:
            # set cost function
            self.cross_entropy = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(logits, output_placeholder)
            )
            tf.scalar_summary('cross_entropy', self.cross_entropy)

        return self.cross_entropy


    def training(self, cross_entropy):
        with tf.name_scope('training') as scope:
            # set learning rete & back-propergation method
            self.train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

        return self.train_step


    def test(self, input_placeholder, output_placeholder, logits, inputs, outputs):
        with tf.name_scope('test') as scope:
            correct_prediction = tf.equal(
                tf.argmax(logits, 1),
                tf.argmax(output_placeholder, 1)
            )
            self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            tf.scalar_summary('accuracy', self.accuracy)

        return self.accuracy.eval(
            feed_dict={input_placeholder  : inputs, 
                       output_placeholder : outputs,
                       self.keep_prob : 1.0}
        )
    