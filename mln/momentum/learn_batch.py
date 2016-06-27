#!/usr/bin/python
# coding=utf-8

# Measure Modules
import numpy as np
import random
# Handmade Modules
import Mln as mln
import dump as dp
import progress as prg
import functions as funcs


def fitting(neuro_obj = None, epoch = 50000, minibatch_size = 1):

    if neuro_obj: nn_obj = neuro_obj
    else        : nn_obj = mln.Mln().make_neuralnet([2, 3, 1], ['sigmoid', 'sigmoid'], eta = 0.15) # XOR fitting

    # use weight decay.
#    nn_obj.use_weight_decay(0.01)     # unlearnable
#    nn_obj.use_weight_decay(0.001)    # unlearnable
    nn_obj.use_weight_decay(0.0001)   # learnable
#    nn_obj.unuse_weight_decay()
    # use momentum.
#    nn_obj.use_momentum(0.1)
    nn_obj.use_momentum(0.5)
#    nn_obj.use_momentum(0.9)
#    nn_obj.unuse_momentum()

    training_data = \
        [\
            (np.array([[0.], [0.]]), np.array([0.])),\
            (np.array([[0.], [1.]]), np.array([1.])),\
            (np.array([[1.], [0.]]), np.array([1.])),\
            (np.array([[1.], [1.]]), np.array([0.])),\
        ]

    # batch learning, such as take small size packaged training data to NN which like a batch sequencially.
    dp.obj_dump(nn_obj, './default-fitting-batch.pkl')

    print "-----weight weight-----:"; print [item.w for item in nn_obj.weight]
    print "-----weight bios-----:"; print [item.b for item in nn_obj.weight]

    mb_size = minibatch_size
    for i in range(0, epoch):
        prg.show_progressxxx(i+1, epoch)

        xxx = random.sample(training_data, 4)
        x = []
        d = []
        for i in range(0, mb_size):
            x.append(xxx[i][0])
            d.append(xxx[i][1])

        nn_obj.batch_learn(x[0:mb_size], d[0:mb_size], mb_size)
    prg.end_progress()

    dp.obj_dump(nn_obj, './learn-fitting-batch.pkl')

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


def binary_classification(neuro_obj = None, epoch = 50000, minibatch_size = 1):
 
    if neuro_obj: nn_obj = neuro_obj
    else        : nn_obj = mln.Mln().make_neuralnet([2, 3, 1], ['sigmoid', 'sigmoid_binary'], eta = 0.15) # XOR classification

    # use weight decay.
#    nn_obj.use_weight_decay(0.01)    # unlearnable
#    nn_obj.use_weight_decay(0.001)   # learnable
    nn_obj.use_weight_decay(0.0001)  # learnable
#    nn_obj.unuse_weight_decay()
    # use momentum.
#    nn_obj.use_momentum(0.1)
    nn_obj.use_momentum(0.5)
#    nn_obj.use_momentum(0.9)
#    nn_obj.unuse_momentum()

    training_data = \
        [\
            (np.array([[0.], [0.]]), np.array([0.])),\
            (np.array([[0.], [1.]]), np.array([1.])),\
            (np.array([[1.], [0.]]), np.array([1.])),\
            (np.array([[1.], [1.]]), np.array([0.])),\
        ]

    # online learning, such as take all training data to NN sequecially.
    dp.obj_dump(nn_obj, './default-binary-classification-batch.pkl')

    print "-----weight weight-----:"; print [item.w for item in nn_obj.weight]
    print "-----weight bios-----:"; print [item.b for item in nn_obj.weight]

    mb_size = minibatch_size
    for i in range(0, epoch):
        prg.show_progressxxx(i+1, epoch)

        xxx = random.sample(training_data, 4)
        x = []
        d = []
        for i in range(0, mb_size):
            x.append(xxx[i][0])
            d.append(xxx[i][1])

        nn_obj.batch_learn(x[0:mb_size], d[0:mb_size], mb_size)
    prg.end_progress()
                
    dp.obj_dump(nn_obj, './learn-binary-classification-batch.pkl')
                
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

 
# If you use this method, then you done 'make_mnist_data.py'.
# So you can make a mnist data & label by pikle with gzip compressed.
def classification(neuro_obj = None, epoch = 100000, minibatch_size = 1):
    num_class = 10

    print 'initialize Neural Network.'
    if neuro_obj: nn_obj = neuro_obj
    else        : nn_obj = mln.Mln().make_neuralnet([28*28, 100, num_class], ['sigmoid', 'softmax'], 0.15) # mnist classification
#    else        : nn_obj = mln.Mln().make_neuralnet([28*28, 1000, num_class], ['sigmoid', 'softmax'], 0.01) # mnist classification

    # use weight decay.
#    nn_obj.use_weight_decay(0.01)   # unlearnable
#    nn_obj.use_weight_decay(0.001)  # learnable
    nn_obj.use_weight_decay(0.0001) # learnable
#    nn_obj.unuse_weight_decay()
    # use momentum.
#    nn_obj.use_momentum(0.1)
    nn_obj.use_momentum(0.5)
#    nn_obj.use_momentum(0.9)
#    nn_obj.unuse_momentum()

    print "dump obj..."
    dp.obj_dump(nn_obj, './default-classification-batch.pkl')

    print 'read training data and label.'
    training_data = dp.obj_load_gzip('../../mnist/mnist-training_all.pkl.gz')

    print 'data      size : ', len(training_data)
    print 'label     size : ', len(training_data)
    print "minibatch size : ", minibatch_size

    data_num = len(training_data)

    print '--start--'
    print '@@ Learn Character Recognition @@'
    mb_size = minibatch_size
    for i in range(0, epoch):
        prg.show_progressxxx(i+1, epoch)

        xxx = random.sample(training_data, data_num)
        x = []
        d = []
        for i in range(0, mb_size):
            x.append(xxx[i][0])
            d.append(xxx[i][1])

        nn_obj.batch_learn(x[0:mb_size], d[0:mb_size], mb_size)
    prg.end_progress()

    print "dump obj..."
    dp.obj_dump(nn_obj, './learn-classification-batch.pkl')
    
    return nn_obj

