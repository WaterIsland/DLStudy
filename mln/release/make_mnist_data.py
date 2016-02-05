#!/usr/bin/python
# coding=utf-8

# Measure Modules
# Handmade Modules
import dump as dp

num_class = 10

print '##### read training data. #####'
training_data  = dp.image_file_read('../../mnist/train-images-idx3-ubyte', normalize = 'max', show = True)

print '##### read training label. #####'
training_label = dp.label_file_read('../../mnist/train-labels-idx1-ubyte', num_class, show = True)

#print '##### compress training data. #####'
#dp.obj_dump_gzip(training_data, '../../mnist/mnist-training_data.pkl.gz')

#print '##### compress training label. #####'
#dp.obj_dump_gzip(training_label, '../../mnist/mnist-training_label.pkl.gz')


print '##### make mnist data & label. #####'
xxx = []
for tdi, tli in zip(training_data, training_label):
    xxx.append([tdi, tli])

print '##### compress training data and label. #####'
dp.obj_dump_gzip(xxx, '../../mnist/mnist-training_all.pkl.gz')



print '##### make test data and label. #####'
test_data  = dp.image_file_read('../../mnist/t10k-images-idx3-ubyte', normalize = 'max', show = True)
dp.obj_dump_gzip(test_data, '../../mnist/mnist-test_data.pkl.gz')
test_label = dp.label_file_read('../../mnist/t10k-labels-idx1-ubyte', num_class, show = True)
dp.obj_dump_gzip(test_label, '../../mnist/mnist-test_label.pkl.gz')
