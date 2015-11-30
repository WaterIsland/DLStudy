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

#x  = np.matrix([[rand.uniform(-0.1, 0.1) for j in range(0, dim[1])] for k in range(0, dim[0])], dtype = 'float')                   
'''
x = np.matrix([[(i+1+4*j) for i in range(0, 4)] for j in range(0,3)])
print x
print x.shape

y = x[0:2, 0:4]
print y
print y.shape
'''
#from sklearn.preprocessing import LabelBinarizer

start = time.time()

# 多層パーセプトロン
learning_count = 50000
neuro_obj = mln.Mln().make_neuralnet([2, 3, 4], ['sigmoid', 'softmax'], 0.01, solved = 'classification')
#neuro_obj = mln.Mln().make_neuralnet([2, 3, 1], ['sigmoid', 'sigmoid'], 0.01, solved = 'classification')
#neuro_obj = mln.Mln().make_neuralnet([2, 3, 1], ['sigmoid', 'sigmoid'], 0.01, solved = 'fit')
neuro_obj.show_element('weight')
neuro_obj.show_element('node')



# Initialization
dp.obj_dump(neuro_obj, './default-br.dump')

# XORの入出力データ
input_data = [[0., 0.], [0.,  1.], [ 1., 0.], [ 1.,  1.]]
#teach_data = [    [0.],      [1.],      [1.],       [0.]]
teach_data = [[0.,0.,0.,1.],[1.,0.,0.,0.],[0.,1.,0.,0.],[0.,0.,1.,0.]]

data_num = len(input_data)


neuro_obj.input_signals(input_data[1]); 
#neuro_obj.show_element('input')
neuro_obj.teach_signals(teach_data[1]); 
#neuro_obj.show_element('teach')
neuro_obj.output_signals(); 
neuro_obj.show_element('output')
neuro_obj.error_signals()
neuro_obj.show_element('err')
neuro_obj.show_element('ttlerr')
exit(0)


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
