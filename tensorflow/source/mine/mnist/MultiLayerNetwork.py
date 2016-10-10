#import sys, os
#import readline
import numpy as np
import scipy as sp
import random
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

import yaml    
import YamlDumper as yd

# With Tensorflow
class MultiLayerNetwork():

    
    #####
    # initialize
    #####
    def __init__(self, fname="default.yaml"):
        self.x = [] # inputs  data placeholder
        self.y = [] # outputs data placeholder
        self.d = [] # teaches data placeholder
        # load NeuralNet configuration file(.yaml)
        self.yd_obj = yd.YamlDumper()
        _ = self.yd_obj.load_yaml(fname)

        
    #####
    # inference
    #####
    def inference(self):
        # parent
        inference_yaml = self.yd_obj.get_parent(element_name='inference')
        network_yaml   = self.yd_obj.get_parent(element_name='network')
        # [inference_yaml]child, depth 1
        layer_num       = self.yd_obj.get_child(network_yaml, element_name='layer_num')
        scope_name_yaml = self.yd_obj.get_child(network_yaml, element_name='scope_name')
        # [network_yaml]child, depth 1
        node_yaml       = self.yd_obj.get_child(inference_yaml, element_name='node_number')
        activation_yaml = self.yd_obj.get_child(inference_yaml, element_name='activation')
        initialize_yaml = self.yd_obj.get_child(inference_yaml, element_name='initialize')
        # [initialize_yaml]child, depth 2
        weight_yaml = self.yd_obj.get_child(initialize_yaml, element_name='weight_name')
        bias_yaml   = self.yd_obj.get_child(initialize_yaml, element_name='bias_name')

        # make inputs placeholder
        node_number = self.yd_obj.get_child(node_yaml, 'layer1')
        scope_names = self.yd_obj.get_child(scope_name_yaml, 'layer1')
        base_name   = self.yd_obj.get_child(scope_names, 'base_name')
        output_name = self.yd_obj.get_child(scope_names, 'output_name')
        with tf.name_scope(base_name) as scope:
            self.x  = tf.placeholder(tf.float32, shape=[None, node_number],  name=output_name)
            self.input_num = node_number

        # make teaches placeholder
        last_node_number = self.yd_obj.get_child(node_yaml, 'layer' + str(layer_num))
        with tf.name_scope('teach') as scope:
            self.d  = tf.placeholder(tf.float32, shape=[None, last_node_number],  name='d')
            self.output_num = last_node_number

        # make inference elements
        x = self.x
        with tf.name_scope('inference') as scope:
            for i in range(1, layer_num):
                # properties of next node
                next_node_number      = self.yd_obj.get_child(node_yaml, 'layer' + str(i+1))
                next_node_scopes      = self.yd_obj.get_child(scope_name_yaml, 'layer' + str(i+1))
                next_node_base_name   = self.yd_obj.get_child(next_node_scopes, 'base_name')
                next_node_output_name = self.yd_obj.get_child(next_node_scopes, 'output_name')
                next_node_activation  = self.yd_obj.get_child(activation_yaml, 'layer' + str(i+1))
                # [name]properties of between present node and next node
                between_node_scopes      = self.yd_obj.get_child(scope_name_yaml, 'layer' + str(i) + '-' + str(i+1))
                between_node_weight_name = self.yd_obj.get_child(between_node_scopes, 'weight_name')
                between_node_bias_name   = self.yd_obj.get_child(between_node_scopes, 'bias_name')
                # [init]properties of between present node and next node
                between_node_initialize = self.yd_obj.get_child(initialize_yaml, 'layer' + str(i) + '-' + str(i+1))
                # weights
                between_node_weight_param = self.yd_obj.get_child(between_node_initialize, 'weight_param')
                is_weight_mean  = False
                is_weight_dev   = False
                is_weight_const = False
                between_node_weight_mean  = 0
                between_node_weight_dev   = 0
                between_node_weight_const = 0
                if 'mean' in between_node_weight_param and 'dev' in between_node_weight_param:
                    between_node_weight_mean  = self.yd_obj.get_child(between_node_weight_param, 'mean')
                    between_node_weight_dev   = self.yd_obj.get_child(between_node_weight_param, 'dev')
                    is_weight_mean = True
                    is_weight_dev  = True
                elif 'const' in between_node_weight_param:
                    between_node_weight_const = self.yd_obj.get_child(between_node_weight_param, 'const')
                    is_weight_const = True
                # biases
                between_node_bias_param   = self.yd_obj.get_child(between_node_initialize, 'bias_param')
                is_bias_mean  = False
                is_bias_dev   = False
                is_bias_const = False
                between_node_bias_mean  = 0
                between_node_bias_dev   = 0
                between_node_bias_const = 0
                if 'mean' in between_node_bias_param and 'dev' in between_node_bias_param:
                    between_node_bias_mean    = self.yd_obj.get_child(between_node_bias_param, 'mean')
                    between_node_bias_dev     = self.yd_obj.get_child(between_node_bias_param, 'dev')
                    is_bias_mean = True
                    is_bias_dev  = True
                elif 'const' in between_node_bias_param:
                    between_node_bias_const   = self.yd_obj.get_child(between_node_bias_param, 'const')
                    is_bias_const = True

                # make weight parameters
                with tf.name_scope(between_node_weight_name) as scope:
                    # make weight and bias
                    if is_weight_mean == True & is_weight_dev == True:
                        w = tf.Variable(
                            tf.random_normal(
                                shape=[node_number, next_node_number],
                                mean=between_node_weight_mean, 
                                stddev=between_node_weight_dev,
                                dtype=tf.float32
                            )
                        )
                    # shouldn't use this. because after Training, you'll get a bad performance.
                    elif is_weight_const == True:
                        w = tf.Variable(
                            tf.constant(
                                between_node_weight_const,
                                shape=[node_number, next_node_number],
                                dtype=tf.float32
                            )
                        )
                    
                # make bias parameters
                with tf.name_scope(between_node_bias_name) as scope:
                    if is_bias_mean == True & is_bias_dev == True:
                        b = tf.Variable(
                            tf.random_normal(
                                shape=[next_node_number], 
                                mean=between_node_bias_mean, 
                                stddev=between_node_bias_dev,
                                dtype=tf.float32
                            )
                        )    
                    elif is_bias_const == True:
                        b = tf.Variable(
                            tf.constant(
                                between_node_bias_const,
                                shape=[next_node_number],
                                dtype=tf.float32
                            )
                        )

                # make outoput evaluation
                with tf.name_scope(next_node_base_name) as scope:
                    if next_node_activation == 'relu':
                        y = tf.nn.relu(tf.matmul(x, w) + b, name=next_node_activation)
                    elif next_node_activation == 'relu6':
                        y = tf.nn.relu6(tf.matmul(x, w) + b, name=next_node_activation)
#                    elif next_node_activation == 'crelu':
#                        y = tf.nn.crelu(tf.matmul(x, w) + b, name=next_node_activation)
                    elif next_node_activation == 'elu':
                        y = tf.nn.elu(tf.matmul(x, w) + b, name=next_node_activation)
                    elif next_node_activation == 'softplus':
                        y = tf.nn.softplus(tf.matmul(x, w) + b, name=next_node_activation)
                    elif next_node_activation == 'softsign':
                        y = tf.nn.softsign(tf.matmul(x, w) + b, name=next_node_activation)
                    elif next_node_activation == 'sigmoid':
                        y = tf.nn.sigmoid(tf.matmul(x, w) + b, name=next_node_activation)
                    elif next_node_activation == 'tanh':
                        y = tf.nn.tanh(tf.matmul(x, w) + b, name=next_node_activation)
                    elif next_node_activation == 'softmax':
                        y = tf.nn.softmax(tf.matmul(x, w) + b, name=next_node_activation)
                    # output layer only
                    if i == (layer_num - 1):
                        self.y = y
                # trick
                node_number = next_node_number
                x = y

                
    #####                
    # loss
    #####
    def loss(self):
        loss_yaml = self.yd_obj.get_parent(element_name='loss')
        function  = self.yd_obj.get_child(loss_yaml, element_name='function')
        
        # make loss elements
        with tf.name_scope('loss') as scope:
            # set cost function
            if function == 'crossentoropy':
                self.cost = tf.reduce_mean(-tf.reduce_sum(self.d * tf.log(self.y), reduction_indices=[1]))
            elif function == 'logistic': # minimize likelihood estimasion(it's a coined word. read minimize as maximize.)
                self.cost = tf.reduce_mean(-tf.reduce_sum(self.d * tf.log(self.y) + (1 - self.d) * tf.log(1 - self.y), reduction_indices=[1]))
            elif function == 'mse': # mean squared error
                self.cost = tf.reduce_mean(tf.square(self.y - self.d))

        # set summaries to tensorflow's summary
        tf.scalar_summary('cross_entropy', self.cost)
    
    
    #####
    # training
    #####
    def training(self):
        training_yaml = self.yd_obj.get_parent(element_name='training')
        function      = self.yd_obj.get_child(training_yaml, element_name='function')
        eta           = self.yd_obj.get_child(training_yaml, element_name='eta')
        optimization  = self.yd_obj.get_child(training_yaml, element_name='optimization')

        with tf.name_scope('training') as scope:
            # set learning rete & back-propergation method
            if function == 'GradientDescentOptimizer':
                self.train_step = tf.train.GradientDescentOptimizer(eta)

            if optimization == 'minimize':
                self.train_step = self.train_step.minimize(self.cost)

            
    #####
    # test(for multi-categorize problems, such as MNIST)
    #####
    def test(self, x, d):
        with tf.name_scope('test') as scope:
            correct_prediction = tf.equal(
                tf.argmax(self.y, 1, name='output_ArgMax'),
                tf.argmax(self.d, 1, name='teach_ArgMax')
            )
            self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

        # set summaries to tensorflow's summary
        tf.scalar_summary('accuracy', self.accuracy)

        # show value of percent that NN can correct prediction
        return self.accuracy.eval(
            feed_dict={self.x: x, 
                       self.d: d})
            
        
    #####
    # get_accuracy
    #####
    def get_accuracy(self, x, d, total_number, use_number):
        num = random.randint(0, total_number)
        accuracy = self.test(
            x[num:num+use_number], 
            d[num:num+use_number]
        )
        
        return accuracy

####
## Test code
#####
if __name__ == '__main__':
    from tensorflow.examples.tutorials.mnist import input_data

    # read mnist data
    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

    # make graph scope
    with tf.Graph().as_default(): 
    
        # make session
        sess = tf.InteractiveSession()

        obj = MultiLayerNetwork(fname='3Layer-NN.yaml')
#        obj = MultiLayerNetwork(fname='4Layer-NN.yaml')
        obj.inference()
        obj.loss()
        obj.training()

        # initialize weight and bias
        sess.run(tf.initialize_all_variables())

        # check NN performance(init)
        accuracy = obj.test(mnist.test.images, mnist.test.labels)
        print("Accuracy[%f]" % accuracy)

        # set all tensorflow's summaries to graph
        summary_op = tf.merge_all_summaries()
        summary_writer = tf.train.SummaryWriter('data', graph=sess.graph)    

        # training
        for step in range(3001):
            # set batch data
            batch = mnist.train.next_batch(20)
            feed_dict={obj.x: batch[0], obj.d: batch[1]}
            # run train
            _, cost, accuracy, summary_str = sess.run([obj.train_step, obj.cost, obj.accuracy, summary_op], feed_dict=feed_dict)
            if step % 20 == 0:
                # output summary to graph
                summary_writer.add_summary(summary_str, step)

                # validation data's accuracy
                validation_data_accuracy = obj.get_accuracy(
                    mnist.validation.images, 
                    mnist.validation.labels,
                    len(mnist.validation.images), 
                    100)

                # test data's accuracy
                test_data_accuracy = obj.get_accuracy(
                    mnist.test.images, 
                    mnist.test.labels,
                    len(mnist.test.images), 
                    100)

                print(
                    "Train Step:[%4d], Accuracy[training : %.3f, validation : %.3f, test : %.3f], Cost[%2.3f]"
                    % (step, accuracy, validation_data_accuracy, test_data_accuracy, cost)
                )
                
                
        # check NN performance(init)
        accuracy = obj.test(mnist.test.images, mnist.test.labels)
        print("Accuracy[%f]" % accuracy)                
                
        # close summary writer
        summary_writer.close()

        # close session
        sess.close()
        
 
    