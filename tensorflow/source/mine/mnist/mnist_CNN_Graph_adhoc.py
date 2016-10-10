#!/usr/local/bin/python
# -*- coding: utf-8 -*-

import sys, os
import readline
import numpy as np
import scipy as sp
import random
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

import time


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
#    print("-----weight : ", initial)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
#    print("$$$$$bias : ", initial)
    return tf.Variable(initial)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)


with tf.Graph().as_default(): 

    sess = tf.InteractiveSession()

    with tf.name_scope('input') as scope:
        x = tf.placeholder(tf.float32, shape=[None, 784], name='x')
    with tf.name_scope('teach') as scope:
        y_ = tf.placeholder(tf.float32, shape=[None, 10], name='d')

    x_image = tf.reshape(x, [-1,28,28,1])
    
    # first convolutional layer
    with tf.name_scope('first_convolutional_layer') as scope:
        W_conv1 = weight_variable([5, 5, 1, 32])
        b_conv1 = bias_variable([32])
        h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
        h_pool1 = max_pool_2x2(h_conv1)
#        print("conv : ", h_conv1)
#        print("pool : ", h_pool1)

    
    # second convolutional layer
    with tf.name_scope('second_convolutional_layer') as scope:
        W_conv2 = weight_variable([5, 5, 32, 64])
        b_conv2 = bias_variable([64])
        h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
        h_pool2 = max_pool_2x2(h_conv2)
#        print("conv : ", h_conv2)
#        print("pool : ", h_pool2)
        
    # Densely Connected Layer
    with tf.name_scope('Densely_Connected_layer') as scope:
        W_fc1 = weight_variable([7 * 7 * 64, 1024])
        b_fc1 = bias_variable([1024])
        h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
    
    # Dropout
    with tf.name_scope('Dropout') as scope:
        keep_prob = tf.placeholder(tf.float32)
        h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
    
    # Readout Layer
    with tf.name_scope('Readout_Layer') as scope:
        W_fc2 = weight_variable([1024, 10])
        b_fc2 = bias_variable([10])
        y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2    

    # Train and Evaluate the Model
    with tf.name_scope('loss') as scope:
        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y_conv, y_))
        tf.scalar_summary('cross_entropy', cross_entropy)
    with tf.name_scope('training') as scope:
        train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
    with tf.name_scope('test') as scope:
        correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        tf.scalar_summary('accuracy', accuracy)

    sess.run(tf.initialize_all_variables())


    # set all tensorflow's summaries to graph
    summary_op = tf.merge_all_summaries()
    summary_writer = tf.train.SummaryWriter('data', graph=sess.graph)    

    for i in range(1001):
#    for i in range(20000):
        start = time.time()

        batch = mnist.train.next_batch(50)
#        train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
        _, summary_str = sess.run([train_step, summary_op], feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
        if i % 50 == 0:
            summary_writer.add_summary(summary_str, i)
            train_accuracy = accuracy.eval(feed_dict={
                x:batch[0], y_: batch[1], keep_prob: 1.0})
            print("step %d, training accuracy %g"%(i, train_accuracy))
        elapsed_time = time.time() - start
        print("elapsed_time[%5d]:%1.3f[sec]" % (i, elapsed_time))

    # adhoc technique
    split_number = 20
    total_number = len(mnist.test.images)
    odd_number = total_number % split_number
    div_number = int((total_number - odd_number) / split_number)
    numbers = [div_number for i in range(split_number)]
    if odd_number > 0:
        numbers.append(odd_number)
        split_number = split_number + 1
    print(numbers)

    total_accuracy = 0
    start_number = 0
    for i in range(split_number):
        local_accuracy = accuracy.eval(feed_dict={x: mnist.test.images[start_number:start_number + numbers[i]], 
                                       y_: mnist.test.labels[start_number:start_number + numbers[i]], 
                                       keep_prob: 1.0})
        total_accuracy = total_accuracy + local_accuracy * numbers[i]
        print("[%5d-%5d]test accuracy[%d]: %.3f" % (start_number, start_number + numbers[i], i, local_accuracy))
        start_number = start_number + numbers[i]

    print("Total Accuracy is %.3f" % (total_accuracy / total_number))
#    print("test accuracy %g"%accuracy.eval(feed_dict={
#        x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))    

    # close summary writer
    summary_writer.close()

    # close session
    sess.close()

        