#!/usr/local/bin/python
# -*- coding: utf-8 -*-
# IPython log file

import random
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


# init values
input_size  = 784
hidden_size = 200
output_size = 10
init_mean   = 0.0
init_dev    = 0.1
eta         = 0.01


def inference(input_placeholder):
    # make weight and bias
    W1 = tf.Variable(
        tf.random_normal(
            [784, hidden_size],
            mean=init_mean, 
            stddev=init_dev,
            dtype=tf.float32))
    b1 = tf.Variable(
        tf.random_normal(
            [hidden_size], 
            mean=init_mean, 
            stddev=init_dev,
            dtype=tf.float32)
    )

    W2 = tf.Variable(
        tf.random_normal(
            [hidden_size, output_size], 
            mean=init_mean, 
            stddev=init_dev,
            dtype=tf.float32)
    )
    b2 = tf.Variable(
        tf.random_normal(
            [output_size], 
            mean=init_mean, 
            stddev=init_dev,
            dtype=tf.float32)
    )

    # make outoput evaluation
#    y1 = tf.nn.sigmoid(tf.matmul(x, W1) + b1)
#    y1 = tf.nn.softplus(tf.matmul(x, W1) + b1)
    y1 = tf.nn.relu(tf.matmul(x, W1) + b1)
    y2 = tf.nn.softmax(tf.matmul(y1, W2) + b2)
    
    return y2


def loss(logits, output_placeholder):
    # set cost function
    cross_entropy = -tf.reduce_sum(output_placeholder * tf.log(logits))
    return cross_entropy

    
def training(cross_entropy):
    # set learning rete & back-propergation method
    train_step = tf.train.GradientDescentOptimizer(eta).minimize(cross_entropy)    
    return train_step


def test(input_placeholder, output_placeholder, logits, inputs, outputs):
    correct_prediction = tf.equal(
        tf.argmax(logits,1),
        tf.argmax(output_placeholder,1)
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
    
    # read mnist data
    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

    # make session
    sess = tf.InteractiveSession()

    # make input and output
    x  = tf.placeholder(tf.float32, shape=[None, input_size])
    y_ = tf.placeholder(tf.float32, shape=[None, output_size])

    # make Tensorflow base elements
    logits = inference(x)
    cross_entropy = loss(logits, y_)
    train_step = train(cross_entropy)

    # initialize weight and bias
    sess.run(tf.initialize_all_variables())

    # check NN performance(init)
    accuracy = test(x, y_, logits, mnist.test.images, mnist.test.labels)
    print("Accuracy[%f]" % accuracy)

    # training
    for i in range(2000):
        batch = mnist.train.next_batch(20)
        train_step.run(feed_dict={x: batch[0], y_: batch[1]})
        if i % 20 == 0:
            num = random.randint(0, len(mnist.validation.images))
            accuracy = test(x,
                            y_, 
                            logits,
                            mnist.validation.images[num:num+100],
                            mnist.validation.labels[num:num+100]
                           )
            print("Train Step:[%d], Accuracy[%f]" % (i, accuracy))

    # check NN performance(finish training)
    accuracy = test(x, y_, logits, mnist.test.images, mnist.test.labels)
    print("Accuracy[%f]" % accuracy)

    # close session
    sess.close()

