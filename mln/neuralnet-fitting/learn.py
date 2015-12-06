#!/usr/bin/python
# coding=utf-8

import Mln as mln
import numpy as np
import progress as prg

'''
n_obj = mln.Mln()
n_obj.make_neuralnet([2, 3, 3], ['sigmoid', 'softmax'])

n_obj.input_signals([1., 0])
n_obj.teach_signals([1., 0., 0.])
print "-----weight weight-----:"; print [item.w for item in n_obj.weight]
print "-----weight bios-----:"; print [item.b for item in n_obj.weight]
print "-----node-----:"; print [item.z for item in n_obj.node]
print "-----teach-----:"; print n_obj.d
n_obj.output_signals()
#print "-----node-----:"; print [item.z for item in n_obj.node]
'''


n_obj = mln.Mln()
n_obj.make_neuralnet([2, 3, 1], ['sigmoid', 'sigmoid'], eta = 0.15)
#n_obj.make_neuralnet([2, 3, 2], ['sigmoid', 'linear'])
#print "-----node-----:"; 
#for item in n_obj.node: print "-----z-----:"; print item.df

print "-----weight weight-----:"; print [item.w for item in n_obj.weight]
print "-----weight bios-----:"; print [item.b for item in n_obj.weight]

training_data = \
    [\
    (np.array([[0.], [0.]]), np.array([0.])),\
    (np.array([[0.], [1.]]), np.array([1.])),\
    (np.array([[1.], [0.]]), np.array([1.])),\
    (np.array([[1.], [1.]]), np.array([0.])),\
    ]

#n_obj.feedforward(training_data[0][0], training_data[0][1])
#print n_obj.node[-1].z
#n_obj.feedforward(training_data[1][0], training_data[1][1])
#print n_obj.node[-1].z
#n_obj.feedforward(training_data[2][0], training_data[2][1])
#print n_obj.node[-1].z
#n_obj.feedforward(training_data[3][0], training_data[3][1])
#print n_obj.node[-1].z

#x = training_data[3][0]
#d = training_data[3][1]
#print "x:"; print x
#print "d:"; print d

#print "-----node-----:"; 
#for item in n_obj.node: print "-----z-----:"; print item.z

epoch = 50000
for i in range(0, epoch):
    prg.show_progressxxx(i+1, epoch)

    for j in range(0, 4):
#        print "%%%%%%%%%%%%%%%%%%%%%%%%%"
#        print training_data[j][0]
#        print training_data[j][1]
#        print "%%%%%%%%%%%%%%%%%%%%%%%%%"
        n_obj.learn(training_data[j][0], training_data[j][1], 1, 1)

prg.end_progress()

n_obj.feedforward(training_data[0][0], training_data[0][1])
print n_obj.node[-1].z
n_obj.feedforward(training_data[1][0], training_data[1][1])
print n_obj.node[-1].z
n_obj.feedforward(training_data[2][0], training_data[2][1])
print n_obj.node[-1].z
n_obj.feedforward(training_data[3][0], training_data[3][1])
print n_obj.node[-1].z

print "@@@@@END@@@@@"
print "-----weight weight-----:"; print [item.w for item in n_obj.weight]
print "-----weight bios-----:"; print [item.b for item in n_obj.weight]

#n_obj.learn(x, d, 1, 1)
#print "-----node-----:"; 
#for item in n_obj.node: print "-----z-----:"; print item.z


