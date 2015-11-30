#!/usr/bin/python
# -*- coding: utf-8 -*-
#
# This data making method refered by
#   http://scikit-learn.org/stable/auto_examples/classification/plot_digits_classification.html
#   https://github.com/sylvan5/PRML/blob/master/ch5/digits.py
#

import time
import numpy as np
import Mln as mln
import dump as dp
import progress as prg

#from sklearn.preprocessing import LabelBinarizer

start = time.time()

# 多層パーセプトロン
learning_count = 50000
neuro_obj = mln.Mln().make_neuralnet([2, 3, 1], ['sigmoid', 'sigmoid'], 0.01, solved = 'classification')

# Initialization
dp.obj_dump(neuro_obj, './default-br.dump')

# XORの入出力データ
input_data = [[0., 0.], [0.,  1.], [ 1., 0.], [ 1.,  1.]]
teach_data = [    [0.],      [1.],      [1.],       [0.]]

data_num = len(input_data)

print '--start--'
print '@@ Learn Character Recognition @@'
for j in range(0, learning_count):
    prg.show_progressxxx(j+1, learning_count)
    
    for i in range(0, 4):
        neuro_obj.learn(input_data[i], teach_data[i])
prg.end_progress()
print ''

dp.obj_dump(neuro_obj, './learn-br.dump')

print("elapsed_time:{0} sec.".format(time.time() - start))
