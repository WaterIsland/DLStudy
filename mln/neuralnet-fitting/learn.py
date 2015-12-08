#!/usr/bin/python
# coding=utf-8

# Measure Modules
import numpy as np
# Handmade Modules
import Mln as mln
import dump as dp
import progress as prg
import functions as funcs


def fitting(neuro_obj = None):
    epoch = 50000

    if neuro_obj: nn_obj = neuro_obj
    else:
        nn_obj = mln.Mln().make_neuralnet([2, 3, 1], ['sigmoid', 'sigmoid'], eta = 0.15) # XOR fitting

    training_data = \
        [\
            (np.array([[0.], [0.]]), np.array([0.])),\
            (np.array([[0.], [1.]]), np.array([1.])),\
            (np.array([[1.], [0.]]), np.array([1.])),\
            (np.array([[1.], [1.]]), np.array([0.])),\
        ]

    dp.obj_dump(nn_obj, './default-fitting.pkl')

    print "-----weight weight-----:"; print [item.w for item in nn_obj.weight]
    print "-----weight bios-----:"; print [item.b for item in nn_obj.weight]

    for i in range(0, epoch):
        prg.show_progressxxx(i+1, epoch)
        
        for j in range(0, 4):
            nn_obj.learn(training_data[j][0], training_data[j][1], 1, 1)
    prg.end_progress()

    dp.obj_dump(nn_obj, './laern-fitting.pkl')

    print "@@@@@TEST@@@@@"
    nn_obj.feedforward(training_data[0][0], training_data[0][1])
    print nn_obj.node[-1].z
    nn_obj.feedforward(training_data[1][0], training_data[1][1])
    print nn_obj.node[-1].z
    nn_obj.feedforward(training_data[2][0], training_data[2][1])
    print nn_obj.node[-1].z
    nn_obj.feedforward(training_data[3][0], training_data[3][1])
    print nn_obj.node[-1].z
    
    print "@@@@@END@@@@@"
    print "-----weight weight-----:"; print [item.w for item in nn_obj.weight]
    print "-----weight bios-----:"; print [item.b for item in nn_obj.weight]

    return nn_obj


def binary_classification(neuro_obj = None):
    epoch = 50000

    if neuro_obj: nn_obj = neuro_obj
    else:
        nn_obj = mln.Mln().make_neuralnet([2, 3, 1], ['sigmoid', 'sigmoid_binary'], eta = 0.15) # XOR classification

    training_data = \
        [\
            (np.array([[0.], [0.]]), np.array([0.])),\
            (np.array([[0.], [1.]]), np.array([1.])),\
            (np.array([[1.], [0.]]), np.array([1.])),\
            (np.array([[1.], [1.]]), np.array([0.])),\
        ]

    dp.obj_dump(nn_obj, './default-binary-classification.pkl')

    print "-----weight weight-----:"; print [item.w for item in nn_obj.weight]
    print "-----weight bios-----:"; print [item.b for item in nn_obj.weight]

    for i in range(0, epoch):
        prg.show_progressxxx(i+1, epoch)
        
        for j in range(0, 4):
            nn_obj.learn(training_data[j][0], training_data[j][1], 1, 1)
    prg.end_progress()

    dp.obj_dump(nn_obj, './laern-binary-classification.pkl')

    print "@@@@@TEST@@@@@"
    nn_obj.feedforward(training_data[0][0], training_data[0][1])
    print nn_obj.node[-1].z
    nn_obj.feedforward(training_data[1][0], training_data[1][1])
    print nn_obj.node[-1].z
    nn_obj.feedforward(training_data[2][0], training_data[2][1])
    print nn_obj.node[-1].z
    nn_obj.feedforward(training_data[3][0], training_data[3][1])
    print nn_obj.node[-1].z
    
    print "@@@@@END@@@@@"
    print "-----weight weight-----:"; print [item.w for item in nn_obj.weight]
    print "-----weight bios-----:"; print [item.b for item in nn_obj.weight]

    return nn_obj

 
def classification(neuro_obj = None):
    num_class = 10
    epoch = 100000

    print 'initialize Neural Network.'
    if neuro_obj: nn_obj = neuro_obj
    else        : nn_obj = mln.Mln().make_neuralnet([28*28, 1000, num_class], ['sigmoid', 'softmax'], 0.01)

    print "dump obj..."
    dp.obj_dump(nn_obj, './default-classification.pkl')

    print 'read training data.'
    training_data  = dp.image_file_read('../../mnist/train-images-idx3-ubyte', normalize = 'max', show = True)
    dp.obj_dump_gzip(training_data, '../../mnist/mnist-training_data.pkl.gz')
    training_data  = dp.obj_load_gzip('../../mnist/mnist-training_data.pkl.gz')

    print 'read training label.'
    training_label = dp.label_file_read('../../mnist/train-labels-idx1-ubyte', num_class, show = True)
    dp.obj_dump_gzip(training_label, '../../mnist/mnist-training_label.pkl.gz')
    training_label = dp.obj_load_gzip('../../mnist/mnist-training_label.pkl.gz')
 
    print 'data  size : ', len(training_data)
    print 'label size : ', len(training_label)

    data_num = len(training_data)
    print '--start--'
    print '@@ Learn Character Recognition @@'
    for j in range(0, epoch):
        prg.show_progressxxx(j+1, epoch)
    
        i = np.random.randint(data_num)
        nn_obj.learn(training_data[i], training_label[i], 1, 1)

    prg.end_progress()

    print "dump obj..."
    dp.obj_dump(nn_obj, './learn-classification.pkl')
    
    return nn_obj

