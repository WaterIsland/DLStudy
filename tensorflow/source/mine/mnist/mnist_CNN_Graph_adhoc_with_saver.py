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
## training only
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

        # initialize weight and bias
        sess.run(tf.initialize_all_variables())

        # check NN performance(init)
        accuracy = cnn_obj.test(x, y_, logits, mnist.test.images[0:100], mnist.test.labels[0:100])
        print("Accuracy[%f]" % accuracy)

        # create saver
        saver = tf.train.Saver()
        
        # set all tensorflow's summaries to graph
        summary_op = tf.merge_all_summaries()
        summary_writer = tf.train.SummaryWriter('/tmp/tf-log', graph=sess.graph)    

        # training
        for step in range(10):
    #    for i in range(20000):
            start = time.time()

            # set batch data
            batch = mnist.train.next_batch(50)
            # run train
            _, summary_str = sess.run([train_step, summary_op], feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
            if step % 100 == 0:
                accuracy = cnn_obj.test(x, y_, logits, batch[0], batch[1])
                print("Accuracy[%f]" % accuracy)
                # output summary to graph
                summary_writer.add_summary(summary_str, step)
                
                saver.save(sess, saver_file_name, global_step=step+1) 

            elapsed_time = time.time() - start
            print("elapsed_time[%5d]:%1.3f[sec]" % (step, elapsed_time))

        # close summary writer
        summary_writer.close()

        # save session
        saver.save(sess, saver_file_name, global_step=step+1)
        
        # close session
        sess.close()

        
        