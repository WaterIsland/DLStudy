

#!/usr/local/bin/python
# -*- coding: utf-8 -*-

#import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import math as mt


class DisplayWeights(object):
    
    
    def __init__(self):
        self.fig = None
    
    
    def make_graph_space(self):
        self.fig = plt.figure()

        
    #
    # make_weights_graph_data
    #
    # If you want to make weights to show then you should use it.
    #
    # column  : number of column; you want to number of splitting.
    # weights : 3-D [hidden_size, input_width, input_height]
    #
    def make_weights_graph_data(self, column, weights):
        # [attributes]number of row & odd number of column -> number of plot row
        row = int(mt.floor(np.shape(weights)[0]/column))
        odd_column = np.shape(weights)[0]%column
        plot_row = row
        if odd_column > 0:
            plot_row = row + 1
            
        weights_width  = np.shape(weights[0])[0]
        weights_height = np.shape(weights[0])[1]

        # [attributes]padding 
        padding_value = np.max(weights)
        padding_size  = 4
        padding_tuple = (padding_size, padding_size)
        
        # [attributes]weight size with padding size
        element_width  = weights_width+padding_size*2
        element_height = weights_height+padding_size*2
        
        pool = []
        for i in range(row):
            tmp = []
            for j in range(column):
                tmp.append(np.pad(weights[i*column+j], (padding_tuple, padding_tuple), 'constant', constant_values=padding_value))
            tmp = np.reshape(tmp, (element_width*column, element_height))
            tmp = np.pad(tmp, padding_tuple, 'constant', constant_values=padding_value)
            tmp = np.transpose(tmp)
            if i == 0:
                pool = tmp
            else:
                pool = np.concatenate((pool, tmp), axis=0)

        # if odd column is exist then add above pool.
        # if we have not enough to fill array then we fill here with same size array.
        if odd_column > 0:
            dummy_tmp = np.ones((weights_width, weights_height))*padding_value
            tmp = []
            for i in range(odd_column):
                tmp.append(np.pad(weights[row*column+i], (padding_tuple, padding_tuple), 'constant', constant_values=padding_value))
            for i in range(column - odd_column):
                tmp.append(np.pad(dummy_tmp, (padding_tuple, padding_tuple), 'constant', constant_values=padding_value))

            tmp = np.reshape(tmp, (element_width*column, element_height))
            tmp = np.pad(tmp, padding_tuple, 'constant', constant_values=padding_value)
            tmp = np.transpose(tmp)
            pool = np.concatenate((pool, tmp), axis=0)

        return pool
            
        
    #
    # show_graph
    #
    # If you want to show weights then you should use it.
    #
    # data  : set return value of 'make_weights_graph_data.'
    #
    def show_graph(self, data):
        attribute = plt.subplot(1, 1, 1)
        attribute.set_xticks([])
        attribute.set_yticks([])
        plt.imshow(data, cmap=plt.cm.gray_r) # grayscale color map
#        plt.imshow(data, cmap=plt.cm.jet) # jet color map
#        plt.imshow(data) # default color map

#        plt.pause(-1) # never time wait, such as plt.show()
        plt.pause(0.1) # few time wait
#        plt.draw() # no wait

