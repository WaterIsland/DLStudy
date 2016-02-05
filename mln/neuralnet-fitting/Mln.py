#!/usr/bin/python
# -*- coding: utf-8 -*-

#
# This code reffered on "http://nnadl-ja.github.io/nnadl_site_ja/index.html".
# Above web site's author is Michael Nielsen.
#

import numpy as np

from neuro_element import NeuroWeight
from neuro_element import NeuroNode

# Multi layer neuralnet
class Mln:

    def __init__(self):
        self.neural_element = []
        self.name_of_element = []
        self.weight = []
        self.node = []

    '''
    def __deepcopy__(self, memo):
        mine = Mln()
        for i in range(0, self.number_of_element): mine.neural_element.append(copy.deepcopy(self.neural_element[i]))
        mine.bios  = self.bios; mine.teach = np.copy(self.teach); mine.error = np.copy(self.error); mine.mse   = self.mse
        mine.learning_rate = self.learning_rate; mine.number_of_element = self.number_of_element; mine.number_of_layer   = self.number_of_layer
        for i in range(0, self.number_of_element): mine.name_of_element.append(self.name_of_element[i])
        return mine
    '''
    
    # Initialization (Not Constructor)
    # network_dims     : [input_layer_dimension, hidden_layer_dimension1, hidden_layer_dimension2, ..., output_layer_fimension]
    # activate_funcion : activate function of hidden layers and output layer neurons such as 'perceptron', 'sigmoid', 'tanh'. See "neuro_function.py".
    # eta              : learning rate which use for back propergation (0 < learning_rate <= 1)
    # bios             : all layer's bios value
    # solved           : 'fitting' or 'classification'
    def make_neuralnet(self, network_dims, activate_function, eta = 0.01, bios = 1.0, solved = 'fitting'):
        # make all layer's weight
        for d1, d2 in zip(network_dims, network_dims[1:]): self.weight.append(NeuroWeight().make_element([d1, d2]))
        # make all layer's node
        function = ['linear']           ; [function.append(func)            for func in activate_function]
        derivative_function = ['linear']; [derivative_function.append(func) for func in activate_function]
        derivative_function[-1] = 'cost_' + activate_function[-1]
        for d1, func , d_func in zip(network_dims, function, derivative_function):
            self.node.append(NeuroNode().make_element(d1, func, d_func, bios))
        # make teach signal
        self.d = np.array(self.node[-1].z)
        # make other parameters
        self.num_layer = len(network_dims)
        self.eta       = eta
        self.solved    = solved
 
        return self

    # show any elements on neural network
    # elemets_name : elament's name which you want to show such as following:
    def show_element(self, element_name):
        pass
                
    # set input signals
    # x : input signals
    def input_signals(self, x):
        self.node[0].u = x
        self.node[0].z = self.node[0].f(self.node[0].u)

    # set teach signals
    # d : teach signals
    def teach_signals(self, d):
        self.d = [[di] for di in d]

    # caliculate output signals
    def output_signals(self):
        for weight, node, next_node in zip(self.weight, self.node, self.node[1:]):
            node.z      = node.f(node.u)
            next_node.u = np.dot(weight.w, node.z) + weight.b

        self.node[-1].z = self.node[-1].f(self.node[-1].u)

    # caliculate err signals
    # call after call those; input_signals(), teach_signals(), output_signals()
    def error_signals(self):
        pass

    def feedforward(self, x, d):
        self.input_signals(x)
        self.teach_signals(d)
        self.output_signals()
#        self.error_signals()

    def back_propergation(self, delta_w, delta_b, d):
        # caliculate cost function deriviation
        delta = self.node[-1].df(self.node[-1].z, d)
        delta_w[-1] += np.dot(delta, self.node[-2].z.transpose())
        delta_b[-1] += delta

        # caliculate update value;
        for i in range(2, self.num_layer):
            de_dz = self.node[-i].df(self.node[-i].z)

            delta = np.dot(self.weight[-i+1].w.T, delta) * de_dz
            delta_w[-i] += np.dot(delta, self.node[-i-1].z.transpose())
            delta_b[-i] += delta

    # learn (using input signals to reach teach signals)
    # x : input signals
    # d : teach signals
    def learn(self, x, d):
        delta_w = [np.zeros(item.w.shape) for item in self.weight]
        delta_b = [np.zeros(item.b.shape) for item in self.weight]
        
        self.feedforward(x, d)
        self.back_propergation(delta_w, delta_b, d)

        # update all layer's weights
        for weight, dw, db in zip(self.weight, delta_w, delta_b):
            weight.w -= self.eta * dw
            weight.b -= self.eta * db

            
    def batch_learn(self, x_vec, d_vec, minibatch_size):
        delta_w = [np.zeros(item.w.shape) for item in self.weight]
        delta_b = [np.zeros(item.b.shape) for item in self.weight]

#        print ""
        for x, d in zip(x_vec, d_vec):
#            delta_w2 = [np.zeros(item.w.shape) for item in self.weight]
#            delta_b2 = [np.zeros(item.b.shape) for item in self.weight]
#            print "x:"; print x
#            print "d:"; print d
            self.feedforward(x, d)
            self.back_propergation(delta_w, delta_b, d)
#            self.back_propergation(delta_w2, delta_b2, d)
#            print "dw"; print delta_w
#            print "dw2"; print delta_w2
#            print "db"; print delta_b
#            print "db2"; print delta_b2
            
        # update all layer's weights
        for weight, dw, db in zip(self.weight, delta_w, delta_b):
            weight.w -= self.eta * dw / minibatch_size
            weight.b -= self.eta * db / minibatch_size
        

    # test (using input signals to reach teach signals)
    # x : input signals
    # d : teach signals
    def test(self, x, d):
        self.feedforward(x, d)
        return [self.get_max_output_index(), self.node[-1].z]

    def add_node(self, add_network_dims, output_funcion = 'sigmoid'):
        pass

    def get_max_output_index(self):
        return np.argmax(self.node[-1].z)

    def get_min_output_index(self):
        return np.argmin(self.node[-1].z)

    def get_output(self):
        return self.node[-1].z

    def get_error(self):
        pass
