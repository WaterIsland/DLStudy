#!/usr/bin/python
# -*- coding: utf-8 -*-

import time
import numpy as np
import Mln as mln
import dump as dp
import progress as prg

start = time.time()

# Initialization
neuro_obj = mln.Mln().make_neuralnet([2, 3, 1], 'tanh', 0.15)
dp.obj_dump(neuro_obj, './default-xor.dump')

# XORの入出力データ
input_data = [[-1., -1.], [-1.,  1.], [ 1., -1.], [ 1.,  1.]]
teach_data = [     [-1.],      [ 1.],      [ 1.],      [-1.]]

learning_count = 50000
data_num = len(input_data)

print '--start--'
print '@@ Learn XOR @@'
for j in range(0, learning_count):
    prg.show_progressxxx(j+1, learning_count)
    for i in range(0, data_num):
        neuro_obj.learn(input_data[i], teach_data[i])    
prg.end_progress()
print ''

dp.obj_dump(neuro_obj, './learn-xor.dump')

print("elapsed_time:{0} sec.".format(time.time() - start))
