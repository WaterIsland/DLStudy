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

import ExtendedTensorflowCNN as extf_cnn


saver_file_name = 'save/mnist-CNN.saver.tf'

######
## test code
######
## test only
######
if __name__ == '__main__':
    
    cnn_obj = extf_cnn.ExtendedTensorflowCNN()

    # read mnist data
    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
    
    # make graph scope
    with tf.Graph().as_default(): 
        # make session
        sess = tf.InteractiveSession()

        # make input and output
        with tf.name_scope('input') as scope:
            x = tf.placeholder(tf.float32, shape=[None, 784], name='x')
            x_image = tf.reshape(x, [-1,28,28,1], name='x-pixel_order')
        with tf.name_scope('teach') as scope:
            y_ = tf.placeholder(tf.float32, shape=[None, 10], name='d')

        # make Tensorflow base elements
        logits        = cnn_obj.inference(x_image)
        cross_entropy = cnn_obj.loss(logits, y_)
        train_step    = cnn_obj.training(cross_entropy)
        # 'keep_prob' is probability of drop out.
        # You shoukd get this value(placeholder) after inference().
        keep_prob     = cnn_obj.keep_prob

        # create saver & restore 
        # if you wanna use new Tensor, then you must use "tf.initialize_variables()" to initialize it.
        saver = tf.train.Saver()
        saver.restore(sess, saver_file_name + "-10")            

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
            local_accuracy = cnn_obj.test(x, 
                                          y_,
                                          logits, 
                                          mnist.test.images[start_number:start_number + numbers[i]], 
                                          mnist.test.labels[start_number:start_number + numbers[i]])
            total_accuracy = total_accuracy + local_accuracy * numbers[i]
            print("[%5d-%5d]test accuracy[%d]: %.3f" % (start_number, start_number + numbers[i], i, local_accuracy))
            start_number = start_number + numbers[i]

        print("Total Accuracy is %.3f" % (total_accuracy / total_number))

#        # save session
#        saver.save(sess, saver_file_name, global_step=step+1)
        
        # close session
        sess.close()
        