#!/usr/bin/python
# -*- coding: utf-8 -*-

import time
import numpy as np
import Mln as mln
import dump as dp
import progress as prg

start = time.time()

# XORの入出力データ
input_data = [[0., 0.], [0.,  1.], [ 1., 0.], [ 1.,  1.]]
teach_data = [    [0.],      [1.],      [1.],       [0.]]



data_num = len(input_data)

print '--start--'
print '@@ Show before learning @@'
neuro_obj = dp.obj_load('./default-br.dump')
for i in range(0, data_num):
    print '------Input[%d]------' % i
    neuro_obj.test(input_data[i], teach_data[i])    
    neuro_obj.show_element('input')
    neuro_obj.show_element('output')
    neuro_obj.show_element('ttlerr')
print ''
    
print '@@ Show after learning @@'
neuro_obj = dp.obj_load('./learn-br.dump')
for i in range(0, data_num):
    print '------Input[%d]------' % i
    neuro_obj.test(input_data[i], teach_data[i])    
    neuro_obj.show_element('input')
    neuro_obj.show_element('output')
    neuro_obj.show_element('ttlerr')
print ''
    
print("elapsed_time:{0} sec.".format(time.time() - start))
