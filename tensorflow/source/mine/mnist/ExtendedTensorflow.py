#!/usr/local/bin/python
# -*- coding: utf-8 -*-

import tensorflow as tf

# init values
input_size  = 784
hidden_size = 200
output_size = 10
init_mean   = 0.0
init_dev    = 0.1
eta         = 0.01

class ExtendedTensorflow():
    
#    def __init__(self):
        
    def inference(self, input_placeholder):
        with tf.name_scope('inference') as scope:
            with tf.name_scope('W1') as scope:
                # make weight and bias
                W1 = tf.Variable(
                    tf.random_normal(
                        [input_size, hidden_size],
                        mean=init_mean, 
                        stddev=init_dev,
                        dtype=tf.float32))
            with tf.name_scope('b1') as scope:
                b1 = tf.Variable(
                    tf.random_normal(
                        [hidden_size], 
                        mean=init_mean, 
                        stddev=init_dev,
                        dtype=tf.float32)
                )

            with tf.name_scope('W2') as scope:
                W2 = tf.Variable(
                    tf.random_normal(
                        [hidden_size, output_size], 
                        mean=init_mean, 
                        stddev=init_dev,
                        dtype=tf.float32)
                )
            with tf.name_scope('b2') as scope:
                b2 = tf.Variable(
                    tf.random_normal(
                        [output_size], 
                        mean=init_mean, 
                        stddev=init_dev,
                        dtype=tf.float32)
                )

            # make outoput evaluation
            with tf.name_scope('y1') as scope:
                y1 = tf.nn.relu(tf.matmul(input_placeholder, W1) + b1)
            #    y1 = tf.nn.sigmoid(tf.matmul(input_placeholder, W1) + b1)
            #    y1 = tf.nn.softplus(tf.matmul(input_placeholder, W1) + b1)
            with tf.name_scope('output') as scope:
                y2 = tf.nn.softmax(tf.matmul(y1, W2) + b2)
       
        return y2


    def loss(self, logits, output_placeholder):
        with tf.name_scope('loss') as scope:
            # set cost function
            cross_entropy = -tf.reduce_sum(output_placeholder * tf.log(logits))

        return cross_entropy


    def training(self, cross_entropy):
        with tf.name_scope('training') as scope:
            # set learning rete & back-propergation method
            train_step = tf.train.GradientDescentOptimizer(eta).minimize(cross_entropy)    

        return train_step


    def test(self, input_placeholder, output_placeholder, logits, inputs, outputs):
        with tf.name_scope('test') as scope:
            correct_prediction = tf.equal(
                tf.argmax(logits, 1),
                tf.argmax(output_placeholder, 1)
            )
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

        # show value of percent that NN can correct prediction
        return accuracy.eval(
            feed_dict={input_placeholder:  inputs, 
                       output_placeholder: outputs})

