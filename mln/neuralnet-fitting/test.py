#!/usr/bin/python
# coding=utf-8


# Measure Modules
import numpy as np
# Handmade Modules
import Mln as mln
import dump as dp
import progress as prg
import functions as funcs


def test_classification():
    num_class = 10

    print 'initialize Neural Network.'
    nn_obj = dp.obj_load('./learn-classification.pkl')

    print 'read test data.'
    test_data  = dp.image_file_read('../../mnist/t10k-images-idx3-ubyte', normalize = 'max', show = True)
    dp.obj_dump_gzip(test_data, '../../mnist/mnist-test_data.pkl.gz')
    test_data  = dp.obj_load_gzip('../../mnist/mnist-test_data.pkl.gz')

    print 'read test label.'
    test_label = dp.label_file_read('../../mnist/t10k-labels-idx1-ubyte', num_class, show = True)
    dp.obj_dump_gzip(test_label, '../../mnist/mnist-test_label.pkl.gz')
    test_label = dp.obj_load_gzip('../../mnist/mnist-test_label.pkl.gz')
 
    print 'data  size : ', len(test_data)
    print 'label size : ', len(test_label)

    data_num = len(test_data)
    prediction_error = []
    prediction_recog = []

    print '--start--'
    print '@@ Test Character Recognition @@'
    for j in range(0, data_num):
        prg.show_progressxxx(j+1, data_num)
    
        num_recog, list_recog = nn_obj.test(test_data[j], test_label[j])

        if test_label[j][num_recog] == 0:
            prediction_error.append((test_label[j], num_recog))

    prg.end_progress()

    count = len(prediction_error)

    for item in prediction_error:
        print "Truth:", np.argmax(item[0]), ", --> But predict:", item[1]

    print ''
    print 'error : ', count, '/', data_num, '(', count * 1.0 / data_num * 100.0, '%)'

    return


test_classification()

