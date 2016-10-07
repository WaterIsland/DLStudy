#!/usr/local/bin/python
# -*- coding: utf-8 -*-

import random
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


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
            #    y1 = tf.nn.sigmoid(tf.matmul(input_placeholder, W1) + b1)
            #    y1 = tf.nn.softplus(tf.matmul(input_placeholder, W1) + b1)
            with tf.name_scope('y1') as scope:
                y1 = tf.nn.relu(tf.matmul(input_placeholder, W1) + b1)
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


######
## test code
######
if __name__ == '__main__':
    
    extf = ExtendedTensorflow()
    
    # read mnist data
    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

    # make graph scope
    with tf.Graph().as_default(): 
    
        # make session
        sess = tf.InteractiveSession()

        # make input and output
        x  = tf.placeholder(tf.float32, shape=[None, input_size],  name="input")
        y_ = tf.placeholder(tf.float32, shape=[None, output_size], name="teach")

        # make Tensorflow base elements
        logits        = extf.inference(x)
        cross_entropy = extf.loss(logits, y_)
        train_step    = extf.training(cross_entropy)

        # initialize weight and bias
        sess.run(tf.initialize_all_variables())

        # check NN performance(init)
        accuracy = extf.test(x, y_, logits, mnist.test.images, mnist.test.labels)
        print("Accuracy[%f]" % accuracy)

        # set summaries to tensorflow's summary
        tf.scalar_summary('cross_entropy', cross_entropy)

        # set all tensorflow's summaries to graph
        summary_op = tf.merge_all_summaries()
        summary_writer = tf.train.SummaryWriter('data', graph=sess.graph)    

        # training
        for step in range(1, 1001):
            # set batch data
            batch = mnist.train.next_batch(20)
            feed_dict={x: batch[0], y_: batch[1]}
            # run train
            sess.run([train_step, cross_entropy], feed_dict=feed_dict)
            if step % 10 == 0:
                accuracy = sess.run(cross_entropy, feed_dict=feed_dict)
                print("CrossEntropy[%4d]:%lf" % (step, accuracy))
                # output summary to graph
                summary_str = sess.run(summary_op, feed_dict=feed_dict)
                summary_writer.add_summary(summary_str, step)
                
        # close summary writer
        summary_writer.close()
        
        # check NN performance(finish training)
        accuracy = extf.test(x, y_, logits, mnist.test.images, mnist.test.labels)
        print("Accuracy[%f]" % accuracy)

        # close session
        sess.close()
