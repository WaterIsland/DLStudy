#!/usr/local/bin/python
# -*- coding: utf-8 -*-

import math
import numpy as np

# Sigmoid function
def sigmoid(x):
    return (1 / (1 + np.exp(-x)))

# Hypubolic tangent function
def tanh(x):
    return np.tanh(x)

# Linear function
def linear(x):
    return x

# Perceptron function
def perceptron(x):
    return (0 if x <= 0 else 1)

# SoftMax function
def softmax(x):
    tmp = np.exp(x)
    return tmp / np.sum(tmp)


'''
Derivated cost function's means dE/du = dE/dz * dz/du.
On this program, dE/dz calls delta.
'''
# Derivative Sigmoid function
def cost_sigmoid_derivative(y, d):
    return (y - d) * y * (1 - y)

# Derivative Sigmoid function(used to binary classification)
def cost_sigmoid_binary_derivative(y, d):
    return (y - d)

# Derivative Hypubolic tangent function
def cost_tanh_derivative(y, d):
    return np.dot((y - d), (1 - np.square(y)))

# Derivative Linear function
def cost_linear_derivative(y, d):
    return (y - d)

# Derivative Perceptron function
def cost_perceptron_derivative(y, d):
    return np.zeros((y.shape[1], y.shape[0]))

# Derivative SoftMax function
def cost_softmax_derivative(y, d):
    return (y - d)


'''
Derivated function's means dE/dz = dE/dz.
'''
# Prime Sigmoid function
def sigmoid_prime(y):
    return y * (1 - y)

# Prime Sigmoid function(used to binary classification)
def sigmoid_binary_prime(y):
    return y * (1 - y)

# Prime Hypubolic tangent function
def tanh_prime(y):
    return 1. - np.square(y)

# Prime Linear function
def linear_prime(y):
    return np.ones((y.shape[1], y.shape[0]))

# Prime Perceptron function
def perceptron_prime(y):
    return np.zeros((y.shape[1], y.shape[0]))
