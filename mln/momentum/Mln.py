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
        self.use_lambda = False
        self.use_mu     = False

    # use weight decay parameter which is called 'lambda' originally.
    # lambda_wd: this value use usally 0.01 - 0.00001.
    def use_weight_decay(self, lambda_wd = 0.0001):
        print "use weight decay [lambda]: ", lambda_wd, "."
        self.use_lambda = True
        self.lambda_wd  = lambda_wd
        if lambda_wd > 0.01: print "[Warning]value of 'weight decay' is too big."
        
    # unuse weight decay parameter which is called 'lambda' originally.
    def unuse_weight_decay(self):
        print "unuse weight decay [lambda]."
        self.use_lambda = False
        self.lambda_wd  = 0.

    # use momentum parameter which is called 'eta' originally.
    # eta_mmntm: this value use usally 0.0 - 1.0.
    def use_momentum(self, mu_mmntm = 0.5):
        print "use momentum [mu]: ", mu_mmntm, "."
        self.use_mu     = True
        self.mu_mmntm   = mu_mmntm
        self.momentum_w = [np.zeros(item.w.shape) for item in self.weight]
        self.momentum_b = [np.zeros(item.b.shape) for item in self.weight]
        if mu_mmntm > 1.0: print "[Warning]value of 'momentum' is too big."
        if mu_mmntm < 0.0: print "[Warning]value of 'momentum' is too small."

##        print "momentum [w]", self.momentum_w
##        print "momentum [b]", self.momentum_b
            
##        print self.use_mu, self.mu_mmntm
        
    # unuse momentum parameter which is called 'eta' originally.
    def unuse_momentum(self):
        print "unuse momentum [mu]."
        self.use_mu   = False
        self.mu_mmntm = 0.
##        print self.use_mu, self.mu_mmntm
                
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
        # use momentum?
        if self.use_mu is True:
##            print ""
##            print "momentum [w]", self.momentum_w
##            print "momentum [b]", self.momentum_b
            tmp_momentum_w = []
            tmp_momentum_b = []
            for weight, dw, db, mw, mb in zip(self.weight, delta_w, delta_b, self.momentum_w, self.momentum_b):
##                print ""
##                print "[weight]", weight.w
                update_w = -self.eta * dw + self.mu_mmntm * mw
                update_b = -self.eta * db + self.mu_mmntm * mb
                tmp_momentum_w.append(update_w)
                tmp_momentum_b.append(update_b)
                weight.w += update_w
                weight.b += update_b
##                print "[weight]", weight.w
##                print "[up w]", update_w
##                print "[up b]", update_b

##            print "momentum [tmp w]", tmp_momentum_w
##            print "momentum [tmp b]", tmp_momentum_b

##            xxx = []
##            yyy = []
##            [xxx, yyy] = self.caliculate_update_values_with_WD(self.weight, delta_w, delta_b, self.momentum_w, self.momentum_b)

##            print "check matrix"
##            for x, y, uw, ub in zip(xxx, yyy, tmp_momentum_w, tmp_momentum_b):
##                print "ww:", x - uw
##                print "wb:", y - ub
##                print "x:", x
##                print "uw:", uw
##                print "y:", y
##                print "ub:", ub

##            quit()
                
            self.momentum_w = tmp_momentum_w
            self.momentum_b = tmp_momentum_b

##            print "momentum [w]", self.momentum_w
##            print "momentum [b]", self.momentum_b
##            quit()
        else:
            for weight, dw, db in zip(self.weight, delta_w, delta_b):
                weight.w -= self.eta * dw
                weight.b -= self.eta * db
            
    def batch_learn(self, x_vec, d_vec, minibatch_size):
        delta_w = [np.zeros(item.w.shape) for item in self.weight]
        delta_b = [np.zeros(item.b.shape) for item in self.weight]

        for x, d in zip(x_vec, d_vec):
            self.feedforward(x, d)
            self.back_propergation(delta_w, delta_b, d)
            
        # update all layer's weights
        # [use] momentum
        if self.use_mu is True:
            tmp_momentum_w = []
            tmp_momentum_b = []
            # [use]weight decay
            if self.use_lambda is True:
                for weight, dw, db, mw, mb in zip(self.weight, delta_w, delta_b, self.momentum_w, self.momentum_b):
##                    print ""
##                    print "[weight]", weight.w
##                    print "[bias]", weight.b
##                    print "[dw]", dw
##                    print "[db]", db
                    update_w = -(self.eta * dw + self.lambda_wd * weight.w) / minibatch_size + self.mu_mmntm * mw
                    update_b = -self.eta * db / minibatch_size + self.mu_mmntm * mb
                    
##                    xxxupdate_w = -self.eta * dw / minibatch_size + self.mu_mmntm * mw
##                    xxxupdate_b = -self.eta * db / minibatch_size + self.mu_mmntm * mb
##                    print "---------"
##                    print update_w
##                    print xxxupdate_w
##                    print update_w - xxxupdate_w
##                    print update_b
##                    print xxxupdate_b
##                    print update_b - xxxupdate_b
##                    print "---------"
                    
                    tmp_momentum_w.append(update_w)
                    tmp_momentum_b.append(update_b)
                    weight.w += update_w
                    weight.b += update_b
##                    print "[weight]", weight.w
##                    print "[bias]", weight.b
##                    print "[up w]", update_w
##                    print "[up b]", update_b

##                print "momentum [tmp w]", tmp_momentum_w
##                print "momentum [tmp b]", tmp_momentum_b
##                quit()

                self.momentum_w = tmp_momentum_w
                self.momentum_b = tmp_momentum_b

            # [unuse]weight decay
            else:
                for weight, dw, db, mw, mb in zip(self.weight, delta_w, delta_b, self.momentum_w, self.momentum_b):
##                    print ""
##                    print "[weight]", weight.w
##                    print "[bias]", weight.b
##                    print "[dw]", dw
##                    print "[db]", db
                    update_w = -self.eta * dw / minibatch_size + self.mu_mmntm * mw
                    update_b = -self.eta * db / minibatch_size + self.mu_mmntm * mb
                    tmp_momentum_w.append(update_w)
                    tmp_momentum_b.append(update_b)
                    weight.w += update_w
                    weight.b += update_b
##                    print "[weight]", weight.w
##                    print "[bias]", weight.b
##                    print "[up w]", update_w
##                    print "[up b]", update_b

##                print "momentum [tmp w]", tmp_momentum_w
##                print "momentum [tmp b]", tmp_momentum_b
##                quit()

                self.momentum_w = tmp_momentum_w
                self.momentum_b = tmp_momentum_b

        # [unuse] momentum
        else:
            for weight, dw, db in zip(self.weight, delta_w, delta_b):
                # [use] weight decay
                if self.use_lambda is True:
                    weight.w -= (self.eta * dw + self.lambda_wd * weight.w) / minibatch_size
                # [unuse] weight decay
                else:
                    weight.w -= self.eta * dw / minibatch_size
                weight.b -= self.eta * db / minibatch_size

    #
    # use Weight Decay
    # time cost of function calling is bigger than if-else.
    def caliculate_update_values_with_WD(self, weight, delta_w, delta_b, minibatch_size):
        for wght, dw, db in zip(self.weight, delta_w, delta_b):
            wght.w -= (self.eta * dw + self.lambda_wd * wght.w) / minibatch_size

    #
    # use MomenTum
    # time cost of function calling is bigger than if-else.
    def caliculate_update_values_with_MT(self, weight, delta_w, delta_b, momentum_w, momentum_b):
        tmp_momentum_w = []
        tmp_momentum_b = []
        for wght, dw, db, mw, mb in zip(weight, delta_w, delta_b, momentum_w, momentum_b):
                update_w = -self.eta * dw + self.mu_mmntm * mw
                update_b = -self.eta * db + self.mu_mmntm * mb
                tmp_momentum_w.append(update_w)
                tmp_momentum_b.append(update_b)
                wght.w += update_w
                wght.b += update_b
                
        return [tmp_momentum_w, tmp_momentum_b]
    
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

    def get_target_index(self, digit):
        return np.where(np.fliplr(np.argsort(np.reshape(self.node[-1].z, (1, 10)))) == digit)[1] + 1

    def get_order_array(self):
        return np.fliplr(np.argsort(np.reshape(self.node[-1].z, (1, 10))))

    def get_output(self):
        return self.node[-1].z

    def get_error(self):
        pass
