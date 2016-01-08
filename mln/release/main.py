#!/usr/bin/python
# coding=utf-8

# Measure Modules
import time
# Handmade Modules
from learn import fitting
from learn import binary_classification
from learn import classification

start = time.time()

#
# XOR fitting
#
nn_obj = fitting()

#
# XOR Classification
#
#nn_obj = binary_classification()

#
# mnist Classification
#
#nn_obj = classification()

end = time.time()

print("elapsed_time:{0} sec.".format(end - start))
