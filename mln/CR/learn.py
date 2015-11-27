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

from sklearn.preprocessing import LabelBinarizer

start = time.time()

# 多層パーセプトロン
learning_count = 100000
#neuro_obj = mln.Mln().make_neuralnet([28*28, 100, 10], ['tanh', 'softmax'], 0.01, solved = 'classification')
#neuro_obj = mln.Mln().make_neuralnet([28*28, 1000, 10], ['tanh', 'softmax'], 0.01, solved = 'classification')
neuro_obj = mln.Mln().make_neuralnet([28*28, 10000, 10], ['tanh', 'softmax'], 0.01, solved = 'classification')

labels     = dp.label_file_read('../../mnist/train-labels-idx1-ubyte', True)
input_data = dp.image_file_read('../../mnist/train-images-idx3-ubyte', True) 
# ピクセルの値を0.0-1.0に規格化
input_data = input_data.astype(np.float64)
input_data /= input_data.max()
print "input max : ", input_data.max()

print 'data  size : ', len(input_data)
print 'label size : ', len(labels)

# 教師信号の数字を1-of-K表記に変換
# 0 => [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
# 1 => [0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
# ...
# 9 => [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
labels_train = LabelBinarizer().fit_transform(labels)

# Initialization
dp.obj_dump(neuro_obj, './default-cr.dump')

data_num = len(input_data)

print '--start--'
print '@@ Learn Character Recognition @@'
for j in range(0, learning_count):
    prg.show_progressxxx(j+1, learning_count)

    i = np.random.randint(data_num)
    neuro_obj.learn(input_data[i], labels_train[i])
prg.end_progress()
print ''

dp.obj_dump(neuro_obj, './learn-cr.dump')

print("elapsed_time:{0} sec.".format(time.time() - start))
