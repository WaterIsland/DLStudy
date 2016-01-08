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
def back_propergation(errs, unit_output, preunit_output, func, learning_rate, solved = 'fitting'):
    if solved == 'fitting' or solved == 'fit':
        if   func == sigmoid    : delta_part = (errs * unit_output * (1 - unit_output))
        elif func == tanh       : delta_part = (errs * (1 - np.square(unit_output)))
        elif func == linear     : delta_part = errs
        elif func == perceptron : delta_part = (errs * 0)

    elif solved == 'classification' or solved == 'class':
        if   func == sigmoid    : delta_part = errs
        elif func == tanh       : pass
        elif func == linear     : delta_part = errs
        elif func == perceptron : pass
        elif func == softmax    : delta_part = errs

    delta = learning_rate * np.dot(preunit_output, delta_part)
    '''
    print "***************************************"
    print "err    :";print errs
    print "unit   :";print unit_output
    print "preunit:";print preunit_output
    print "func   :";print func
    print "rate   :";print learning_rate
    print "solve  :";print solved
    print "delta-p:";print delta_part
    print "delta  :";print delta
    print "***************************************"
    '''
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


