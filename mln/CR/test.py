#!/usr/bin/python
# -*- coding: utf-8 -*-
#
# This precision method refered by
#   http://scikit-learn.org/stable/auto_examples/classification/plot_digits_classification.html
#   https://github.com/sylvan5/PRML/blob/master/ch5/digits.py
#


import time
import numpy as np
import Mln as mln
import dump as dp
import progress as prg

from sklearn.datasets import load_digits
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import confusion_matrix, classification_report

start = time.time()

input_data = dp.image_file_read('../../mnist/t10k-images-idx3-ubyte', True)
labels     = dp.label_file_read('../../mnist/t10k-labels-idx1-ubyte', True)
# ピクセルの値を0.0-1.0に正規化
input_data = input_data.astype(np.float64)
input_data /= input_data.max()

print 'data  size : ', len(input_data)
print 'label size : ', len(labels)

# 教師信号の数字を1-of-K表記に変換
# 0 => [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
# 1 => [0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
# ...
# 9 => [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
teach_data = LabelBinarizer().fit_transform(labels)

data_num = len(input_data)

print '--start--'
print '@@ Show after learning @@'
neuro_obj = dp.obj_load('./learn-cr.dump')
# テストデータを用いて予測精度を計算
predictions = []
count = 0
for i in range(0, data_num):
#for i in range(0, 100):
    neuro_obj.test(input_data[i], teach_data[i])
    output = neuro_obj.get_output()
    # 予測結果をプール
    predictions.append(neuro_obj.get_max_output_index())
    if neuro_obj.get_max_output_index() != labels[i]:
#    if neuro_obj.get_min_output_index() != labels[i]:
        print '------Input[%d]------' % i, 'miss : ', labels[i], ' -> ', neuro_obj.get_max_output_index()
        count += 1
    else:
        pass
#        print '------Input[%d]------' % i, 'same : ', labels[i], ' -> ', neuro_obj.get_max_output_index()

print ''
print 'error : ', count, '/', data_num, '(', count * 1.0 / data_num * 100.0, '%)'
#print confusion_matrix(labels, predictions)
#print classification_report(labels, predictions)

print("elapsed_time:{0} sec.".format(time.time() - start))
'''
# 誤認識したデータのみ描画
# 誤認識データ数と誤っているテストデータのidxを収集
cnt = 0
error_idx = []
for idx in range(len(labels)):
    if labels[idx] != predictions[idx]:
        print "error: %d : %d => %d" % (idx, labels[idx], predictions[idx])
        error_idx.append(idx)
        cnt += 1

# 描画
import pylab
for i, idx in enumerate(error_idx):
    pylab.subplot(cnt/5 + 1, 5, i + 1)
    pylab.axis('off')
    pylab.imshow(input_data[idx].reshape((8, 8)), cmap=pylab.cm.gray_r)
    pylab.title('%d : %i => %i' % (idx, labels[idx], predictions[idx]))
pylab.show()
'''
