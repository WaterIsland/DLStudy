#!/usr/local/bin/python
# -*- coding: utf-8 -*-

########### Attension
# [n]     means n-dimension array
# [m x n] means (m x n)-dimension array
# no []   means 1-dimension data

import math
import numpy as np

# Back Propergation
#
#
#
def back_propergation(errs, cur_outputs, pre_outputs, func, learning_rate, solved = 'fitting'):
    if solved == 'fitting':
        if   func == sigmoid    : delta_part = (errs * cur_outputs * (1 - cur_outputs))
        elif func == tanh       : delta_part = (errs * (1 - np.square(cur_outputs)))
        elif func == linear     : delta_part = errs
        elif func == perceptron : delta_part = (errs * 0)

    elif solved == 'classification':
        if   func == sigmoid    : delta_part = errs
        elif func == tanh       : pass
        elif func == linear     : pass
        elif func == perceptron : pass
        elif func == softmax    : delta_part = errs
         
    delta = learning_rate * np.dot(pre_outputs, delta_part)
    return [delta, delta_part]

# Sigmoid function
# src_data : Value of input signal
def sigmoid(src_data):
    return (1 / (1 + math.exp(-src_data)))

# Hypubolic tangent function
# src_data : Value of input signal
def tanh(src_data):
    return math.tanh(src_data)

# Linear function
# src_data : Value of input signal
def linear(src_data):
    return src_data

# Perceptron function
# src_data : Value of input signal
def perceptron(src_data):
    return (0 if src_data <= 0 else 1)

# SoftMax function
# src_data : Value of input signal
# sum_data : Sum Value of all input signal
def softmax(src_data):
    tmp = np.exp(src_data)
    return tmp / np.sum(tmp)


