#!/usr/bin/python
# -*- coding: utf-8 -*-

try   : import cPickle as pickle
except: import pickle

import numpy as np
import copy
import progress as prg

def obj_dump(obj, fname = 'dump.dump'):
    with open(fname, 'wb') as file: pickle.dump((obj), file)
        
def obj_load(fname = 'dump.dump'):
    with open(fname, 'rb') as file: obj = pickle.load(file)
    return obj

# fname    : File name
def label_file_read(fname, show = False):
    # set number labels
    with open(fname, 'rb') as file: allLines = file.read()
#    data_size = len(allLines) - 8; label = np.array(np.zeros(data_size), dtype = np.uint8)
    data_size = len(allLines) - 8; label = np.array(np.zeros(data_size), dtype = 'double')
    num_of_item = 0
    for item in allLines[4:8]: num_of_item = (num_of_item << 8 | ord(item))        
    for i in range(0, num_of_item): label[i] = ord(allLines[i+8])

    if show:
        magic_number = 0
        for item in allLines[0:4]: magic_number = (magic_number << 8 | ord(item))
        # count how many numbers of number of label are exist.
        items = [0 for i in range(0, 10)]
        for item in allLines[8:len(allLines)]: items[ord(item)] += 1
        print 'File Size : '             , len(allLines), 'Byte.'
        print 'Magic Number(4Byte) : '   , hex(magic_number)
        print 'Number Of Items(4Byte) : ', num_of_item
        print 'Data size : '             , data_size, 'Byte.'
        for i in range(0, len(items)): print 'Number Of Labels To "%d" : ' %i, items[i]
    return label

# fname    : File name
def image_file_read(fname, show = False):
    # set number labels
    with open(fname, 'rb') as file: allLines = file.read()
    data_size = len(allLines) - 16;
    num_of_item = 0
    for item in allLines[ 4: 8]: num_of_item = (num_of_item << 8 | ord(item))        
    row_of_item = 0
    for item in allLines[ 8:12]: row_of_item = (row_of_item << 8 | ord(item))
    col_of_item = 0
    for item in allLines[12:16]: col_of_item = (col_of_item << 8 | ord(item))        

#    image = np.array(np.zeros((col_of_item*row_of_item*num_of_item)), dtype = np.uint8)
    image = np.array(np.zeros((col_of_item*row_of_item*num_of_item)), dtype = 'double')
    for i in range(0, row_of_item*col_of_item*num_of_item):
        image[i] = ord(allLines[i+16])
    image = image.reshape(num_of_item, row_of_item*col_of_item)

    if show:
        magic_number = 0
        for item in allLines[ 0: 4]: magic_number = (magic_number << 8 | ord(item))
        print 'File Size : ', len(allLines), 'Byte.'
        print 'Magic Number(4Byte) : '     , hex(magic_number)
        print 'Number Of Items(4Byte) : '  , num_of_item
        print 'Number Of Rows(4Byte) : '   , row_of_item
        print 'Number Of Colmuns(4Byte) : ', col_of_item
        print 'Data size : ', data_size, 'Byte.'

    return image


##################################################################
# CAUSION : create many data file, and you spent on waiting.
##################################################################
# fname    : File name
def image_file_dump(fname, show = False):
    # set number labels
    with open(fname, 'rb') as file: allLines = file.read()
    num_of_item = 0
    for item in allLines[ 4: 8]: num_of_item = (num_of_item << 8 | ord(item))        
    row_of_item = 0
    for item in allLines[ 8:12]: row_of_item = (row_of_item << 8 | ord(item))
    col_of_item = 0
    for item in allLines[12:16]: col_of_item = (col_of_item << 8 | ord(item))        

    for i in range(0, num_of_item):
        prg.show_progress(i, num_of_item-1)
        offset = 16
        min_coodinate = i    *row_of_item*col_of_item+offset
        max_coodinate = (i+1)*row_of_item*col_of_item+offset
        obj_dump(allLines[min_coodinate:max_coodinate], fname + str(i) + '.dump')

        if show:
            for j in range(0, row_of_item*col_of_item):
                if  ord(allLines[j+i*row_of_item*col_of_item+16]) > 0.: print '#',
                else                                                  : print ' ',
                if j%row_of_item == 0                                 : print ''
    prg.end_progress()

    
#
# return image of n for file
#
def get_image_n(fname, n):
    # set number labels
    with open(fname, 'rb') as file: allLines = file.read()
    num_of_item = 0
    for item in allLines[ 4: 8]: num_of_item = (num_of_item << 8 | ord(item))        
    row_of_item = 0
    for item in allLines[ 8:12]: row_of_item = (row_of_item << 8 | ord(item))
    col_of_item = 0
    for item in allLines[12:16]: col_of_item = (col_of_item << 8 | ord(item))        

    offset = 16
    min_coodinate = (n-1)*row_of_item*col_of_item+offset
    max_coodinate = n    *row_of_item*col_of_item+offset

    return np.array([ord(item) for item in allLines[min_coodinate:max_coodinate]], dtype = 'double')



        
        
        
