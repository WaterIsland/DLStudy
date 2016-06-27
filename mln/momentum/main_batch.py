#!/usr/bin/python
# coding=utf-8

# Measure Modules
import time
# Handmade Modules
from learn_batch import fitting
from learn_batch import binary_classification
from learn_batch import classification

start = time.time()

#
# XOR fitting
#
#nn_obj = fitting(epoch = 50000, minibatch_size = 2) # mini batch
#nn_obj = fitting(epoch = 20000, minibatch_size = 4) # batch

#
# XOR Classification
#
#nn_obj = binary_classification(epoch = 20000, minibatch_size = 2) # mini batch
#nn_obj = binary_classification(epoch = 20000, minibatch_size = 4) # batch

#
# mnist Classification
#
nn_obj = classification(epoch = 10000, minibatch_size = 10) # mini batch
#nn_obj = classification(epoch = 20000, minibatch_size = 10) # mini batch
#nn_obj = classification(epoch = 20000, minibatch_size = 60000) # batch, using many many time

end = time.time()

print("elapsed_time:{0} sec.".format(end - start))
