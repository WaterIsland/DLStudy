#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import numpy.random as nprndm
import neuro_function as nf

# Weights
class NeuroWeight():
    def __init__(self): pass
        
    def make_element(self, dim):
        if len(dim) == 2:
            self.w = np.array(nprndm.uniform(-0.1, 0.1, (dim[1], dim[0])))
            self.b = np.array(nprndm.uniform(-0.1, 0.1, (dim[1], 1)))
            # Following init value used by Michael Nielsen, reffered on "http://nnadl-ja.github.io/nnadl_site_ja/index.html".
#            self.w = np.random.randn(dim[1], dim[0])
#            self.b = np.random.randn(dim[1], 1)

        else:
            print "Invallid number of element: dim is not 2-dimension."; exit(0)            

        return self
    '''
    def __deepcopy__(self, memo):
        mine = NeuroWeight(); mine.value = copy.deepcopy(self.value); mine.size  = copy.deepcopy(self.size); 
        return mine
        '''

class NeuroNode():
    def __init__(self): pass

    def make_element(self, dim, func = 'sigmoid', derivative_func = 'sigmoid', bios = 1.0):
        self.u = np.array(np.zeros((dim, 1)))
        self.z = np.array(np.zeros((dim, 1)))
        self.bios = bios

        if   func == 'sigmoid'   : self.f = nf.sigmoid
        elif func == 'binary'    : self.f = nf.sigmoid
        elif func == 'tanh'      : self.f = nf.tanh
        elif func == 'linear'    : self.f = nf.linear
        elif func == 'perceptron': self.f = nf.perceptron
        elif func == 'softmax'   : self.f = nf.softmax

        if   derivative_func == 'sigmoid'        : self.df = nf.sigmoid_prime
        elif derivative_func == 'binary'         : self.df = nf.sigmoid_binary_prime
        elif derivative_func == 'tanh'           : self.df = nf.tanh_prime
        elif derivative_func == 'linear'         : self.df = nf.linear_prime
        elif derivative_func == 'perceptron'     : self.df = nf.perceptron_prime
#        elif derivative_func == 'softmax'        : self.df = nf.softmax_prime
        elif derivative_func == 'cost_sigmoid'   : self.df = nf.cost_sigmoid_derivative
        elif derivative_func == 'cost_binary'    : self.df = nf.cost_sigmoid_binary_derivative
        elif derivative_func == 'cost_tanh'      : self.df = nf.cost_tanh_derivative
        elif derivative_func == 'cost_linear'    : self.df = nf.cost_linear_derivative
        elif derivative_func == 'cost_perceptron': self.df = nf.cost_perceptron_derivative
        elif derivative_func == 'cost_softmax'   : self.df = nf.cost_softmax_derivative

        return self

    '''
    def __deepcopy__(self, memo): 
        mine = NeuroNode(); mine.value = copy.deepcopy(self.value); mine.size  = copy.deepcopy(self.size); mine.func  = copy.deepcopy(self.func); 
        return mine
        '''
