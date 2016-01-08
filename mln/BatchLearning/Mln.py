#!/usr/bin/python
# -*- coding: utf-8 -*-

########### Attension
# [n]     means n-dimension array
# [m x n] means (m x n)-dimension array
# no []   means 1-dimension data

import random as rand
import numpy as np
import numpy.linalg as nplnlg
import neuro_function as nf

class NeuroElement:
    def __init__(self)   : self.size = []; self.value = []
    def make_elment(self): return self

# Weights
class NeuroWeight(NeuroElement):
    def __init__(self): NeuroElement().__init__()
        
    def make_elment(self, dim):
        if len(dim) == 2: self.value = np.matrix([[rand.uniform(-0.1, 0.1) for j in range(0, dim[1])] for k in range(0, dim[0])], dtype = 'float')
#        if len(dim) == 2: self.value = np.matrix([[rand.uniform(-0.5, 0.5) for j in range(0, dim[1])] for k in range(0, dim[0])], dtype = 'float')
        self.size = np.array(dim)
        return self

    def __deepcopy__(self, memo):
        mine = NeuroWeight(); mine.value = copy.deepcopy(self.value); mine.size  = copy.deepcopy(self.size); 
        return mine

class NeuroNode(NeuroElement):
    def __init__(self): NeuroElement().__init__(); self.func = 'sigmoid'

    def make_elment(self, dim, func = 'sigmoid'):
        if len(dim) == 1: self.value = np.array([np.zeros(dim)], dtype = 'float')
        if   func == 'sigmoid'   : self.func = nf.sigmoid
        elif func == 'tanh'      : self.func = nf.tanh
        elif func == 'linear'    : self.func = nf.linear
        elif func == 'perceptron': self.func = nf.perceptron
        elif func == 'softmax'   : self.func = nf.softmax
        self.size = np.array(dim)
        return self

    def __deepcopy__(self, memo): 
        mine = NeuroNode(); mine.value = copy.deepcopy(self.value); mine.size  = copy.deepcopy(self.size); mine.func  = copy.deepcopy(self.func); 
        return mine

# Multi layer neuralnet
class Mln:

    def __init__(self):
        # neural_element : This means that input node, weights, hidden node, weights, ..., outputs node are existed sequencially.
        self.neural_element = [] ; self.bios = 0.0           ; self.teach = []         ; self.error = []           ; self.mse = 0.0; 
        self.learning_rate = 0.15; self.number_of_element = 0; self.number_of_layer = 0; self.name_of_element = []
    
    def __deepcopy__(self, memo):
        mine = Mln()
        for i in range(0, self.number_of_element): mine.neural_element.append(copy.deepcopy(self.neural_element[i]))
        mine.bios  = self.bios; mine.teach = np.copy(self.teach); mine.error = np.copy(self.error); mine.mse   = self.mse
        mine.learning_rate = self.learning_rate; mine.number_of_element = self.number_of_element; mine.number_of_layer   = self.number_of_layer
        for i in range(0, self.number_of_element): mine.name_of_element.append(self.name_of_element[i])
        return mine
    
    # Initialization (Not Constructor)
    # network_dims  : [input_layer_dimension, hidden_layer_dimension1, hidden_layer_dimension2, ..., output_layer_fimension]
    # output_funcion : output function of all neurons such as 'perceptron', 'sigmoid', 'tanh'
    # learning_rate  : learning rate which use for back propergation (0 < learning_rate <= 1)
    # bios : all layer's bios value
    def make_neuralnet(self, network_dims, output_function = ['sigmoid'], learning_rate = 0.15, bios = 1.0, solved = 'fitting'):
        if len(output_function) + 1 != len(network_dims): print "Invallid number of element: network_dims and output_function is not same."; exit(0)
        self.learning_rate = learning_rate; self.bios = bios
        # make number of element & layer
        self.number_of_layer = len(network_dims); self.number_of_element = len(network_dims)*2-1;
        # make nodes and weights of all layers
        func = ['linear'] + [func for func in output_function]
        for i in range(0, self.number_of_layer): self.neural_element.append(NeuroNode().make_elment([network_dims[i]], func[i]))
        for i in range(0, self.number_of_layer-1): self.neural_element.insert(i*2+1, NeuroWeight().make_elment(np.array(network_dims[i:i+2])))
        # elements_names presents that a element of parameter of 'neural_elements' is 'node', 'weight', 'error'.
        for i in range(0, self.number_of_layer): self.name_of_element.append('node'); self.name_of_element.append('weight')
        self.name_of_element.pop()
        # make teach & error signals
        self.teach = np.matrix([np.zeros(network_dims[len(network_dims)-1])]); self.error = np.matrix([np.zeros(network_dims[len(network_dims)-1])])
        self.solved = solved
        self.total_error = 0.
        return self

    # show any elements on neural network
    # elemets_name : elament's name which you want to show such as following:
    #       'function': show output functions
    #       'teach'   : show teach signals
    #       'weight'  : show weights
    #       'node'    : show node signals
    #       'err'     : show error signals
    #       'ttlerr'  : show total error signals, if you select "solved = 'fitting'" then you can see MSE, and "solved = 'classification'" then you can see total error.
    def show_element(self, element_name):
        if   element_name == 'teach'   : print '// Teaches            //' , self.teach; return
        elif element_name == 'err'     : print '// Output Errors      //' , self.error; return
        elif element_name == 'ttlerr'  : print '// Output Total Error //' , self.total_error  ; return
        elif element_name == 'input'   : print '// Input              //' , self.neural_element[0].value  ; return
        elif element_name == 'output'  : print '// Output             //' , self.neural_element[len(self.neural_element)-1].value  ; return
        elif element_name == 'weight'  : print '// Bios of all Layer  //' , self.bios
        names = ['Input'] + ['Hidden' for i in range(0, self.number_of_layer-2)] + ['Output']
        for i in range(0, len(self.name_of_element)):
            if element_name == 'weight'  and self.name_of_element[i] == 'weight': 
                messages = '// Weights of ' + names[0]    ; del names[0]; print messages + ' to ' + names[0] + ' Layer //'; print self.neural_element[i].value  
            elif element_name == 'node'  and self.name_of_element[i] == 'node'  : 
                messages = '// Nodes of ' + names[0]      ; del names[0]; print messages + ' //'                          ; print self.neural_element[i].value
            elif element_name == 'func' and self.name_of_element[i] == 'node'   : 
                messages = '// Functions of ' + names[0]  ; del names[0]; print messages + ' //'                          ; print self.neural_element[i].func
                
    # set input signals
    # input_data : input signals, same dimeision as number of input layer's node.
    def input_signals(self, input_data):
        # set input signals to input layer's node
        tmp_data = np.matrix(input_data)
        if self.neural_element[0].value.shape == tmp_data.shape: self.neural_element[0].value = tmp_data
        else: print 'Invallid size of input data:', tmp_data.shape; print 'You must use size of input data:', self.neural_element[0].value.shape

    # set teach signals
    # teach_data : teach signals, same dimeision as number of output layer's node.
    def teach_signals(self, teach_data):
        # set teach signals to output layer's node
        tmp_data = np.matrix(teach_data)
        if self.teach.shape == tmp_data.shape: self.teach = tmp_data
        else: print 'Invallid size of teach data:', tmp_data.shape; print 'You must use size of teach data:', self.teach.shape

    # caliculate err signals
    # call after call those; input_to_neuron(), teach_to_neuron(), output_for_neuron()
    def error_signals(self):
        # solved problem
        if self.solved == 'fitting':
            self.error       = self.neural_element[self.number_of_element-1].value - self.teach
            self.total_error = np.sum(self.error)
        elif self.solved == 'classification':
            # softmax
            if self.neural_element[self.number_of_element-1].func == nf.softmax:
                self.error       = np.array(self.teach) * np.array(np.log(self.neural_element[self.number_of_element-1].value))
                self.total_error = np.sum(self.error)
            # sigmoid
            elif self.neural_element[self.number_of_element-1].func == nf.sigmoid:
                self.error       = np.array(self.teach.T) *np.array(np.log(self.neural_element[self.number_of_element-1].value)) + np.array(1 - self.teach.T) * np.array(np.log(1 - self.neural_element[self.number_of_element-1].value))
                self.total_error = np.sum(self.error)
            # tanh
            elif self.neural_element[self.number_of_element-1].func == nf.tanh:
                pass

    # caliculate output signals
    def output_signals(self):
        tmp_output = self.neural_element[0].value + self.bios
        for i in range(1, self.number_of_element):
            if   self.name_of_element[i] == 'weight': tmp_output = np.dot(tmp_output, self.neural_element[i].value);
            elif self.name_of_element[i] == 'node'  : 
                # softmax function or not
                if self.neural_element[i].func != nf.softmax:
                    tmp_output = np.matrix([self.neural_element[i].func(element + self.bios) for element in np.nditer(tmp_output)]).reshape(tmp_output.shape);
                    self.neural_element[i].value = tmp_output
                else:
                    # add neuron outputs with bios
                    tmp_output = tmp_output + self.bios;
                    # apply softmax function to upper value
                    tmp_output = self.neural_element[i].func(tmp_output)
                    self.neural_element[i].value = tmp_output

    def caliculate_backpropergation(self, de_dy):
        errs = de_dy

        # make update value pool
        delta_pool = [[]]
        for i in range(1, self.number_of_element):
            if self.name_of_element[i] == 'weight': delta_pool.append([np.matrix(np.zeros(self.neural_element[i].value.shape))])
            else                                  : delta_pool.append([])
            
        # caliculate on output layer
        [delta_pool[self.number_of_element-2], delta_part] = \
            nf.back_propergation(\
            errs, \
                np.array(self.neural_element[self.number_of_element-1].value), \
                self.neural_element[self.number_of_element-3].value.T, \
                self.neural_element[self.number_of_element-1].func,\
                self.learning_rate,\
                solved = self.solved);
        errs = np.array(np.dot(delta_part, self.neural_element[self.number_of_element-2].value.T));

        for i in range(self.number_of_element-3, 1, -2):
            [delta_pool[i-1], delta_part] = \
                nf.back_propergation(\
                errs, \
                    np.array(self.neural_element[i].value), \
                    self.neural_element[i-2].value.T, \
                    self.neural_element[i].func, \
                    self.learning_rate, \
                    solved = 'fitting')
            errs = np.array(np.dot(delta_part, self.neural_element[i-1].value.T));
            
        return delta_pool


    def caliculate_de_dy(self, teach_data):
        # caliculate de/dy
        # solved problem
        if self.solved == 'fitting':
            errs = -np.array(self.neural_element[self.number_of_element-1].value - teach_data)
        elif self.solved == 'classification':
            # softmax
            if self.neural_element[self.number_of_element-1].func == nf.softmax:
                errs = np.array(self.neural_element[self.number_of_element-1].value - teach_data)
            # sigmoid
            elif self.neural_element[self.number_of_element-1].func == nf.sigmoid:
                errs = np.array(self.neural_element[self.number_of_element-1].value - teach_data)
            # tanh
            elif self.neural_element[self.number_of_element-1].func == nf.tanh:
                pass

        return errs
    
    # learn (using input signals to reach teach signals)
    # input_data : input signals, same dimeision as number of input layer's node.
    # teach_data : teach signals, same dimeision as number of output layer's node.
    def learn(self, input_data, teach_data):
        # set input and teach signals, and caliculate output and err signals.
        self.input_signals(input_data); self.teach_signals(teach_data); self.output_signals(); self.error_signals()

        errs = self.caliculate_de_dy(teach_data)
        delta_pool = self.caliculate_backpropergation(errs)

        # update all weights
        for i in range(1, self.number_of_element): 
            if self.name_of_element[i] == 'weight': self.neural_element[i].value -= delta_pool[i]


    def batch_learn(self, input_data_set, teach_data_set, n):
        '''
        # set input and teach signals, and caliculate output and err signals.
        self.input_signals(input_data_set[0]); self.teach_signals(teach_data_set[0]); self.output_signals(); self.error_signals()
        err_pool = self.caliculate_de_dy(teach_data_set[0])

#        print "err:"; print err_pool
        for i in range(1, n):
            self.input_signals(input_data_set[i]); self.teach_signals(teach_data_set[i]); self.output_signals(); self.error_signals()
            err_pool += self.caliculate_de_dy(teach_data_set[i])
#            print "Add err:"; print self.caliculate_de_dy(teach_data_set[i])
#            print "AFTER err:"; print err_pool

        delta_pool = self.caliculate_backpropergation(err_pool)
        
        # update all weights
        for i in range(1, self.number_of_element): 
            if self.name_of_element[i] == 'weight': 
#                print "BEFORE:"; print self.neural_element[i].value
#                print "ADD   :"; print delta_pool[i]
#                print "ADD/n :"; print delta_pool[i] / n
                self.neural_element[i].value -= delta_pool[i] / n
#                self.neural_element[i].value -= delta_pool[i]
#                print "AFTER :";print self.neural_element[i].value
        '''


        # set input and teach signals, and caliculate output and err signals.
        self.input_signals(input_data_set[0]); self.teach_signals(teach_data_set[0]); self.output_signals(); self.error_signals()
        err_pool = self.caliculate_de_dy(teach_data_set[0])
        delta_pool = self.caliculate_backpropergation(err_pool)

#        print "------delta_pool [0]------"; 
#        for i in delta_pool:
#            print i

        for i in range(1, n):
            self.input_signals(input_data_set[i]); self.teach_signals(teach_data_set[i]); self.output_signals(); self.error_signals()
            err_pool = self.caliculate_de_dy(teach_data_set[i])
            delta = self.caliculate_backpropergation(err_pool)
            for i in range(0, len(delta)):
                delta_pool[i] += delta[i]
#            print "------Add delta_pool [%d]------" % i; 
#            for item in delta:
#                print item
#            print "------delta_pool [%d]------" % i; 
#            for item in delta_pool:
#                print item

        # update all weights
        for i in range(1, self.number_of_element): 
            if self.name_of_element[i] == 'weight': 
#                print "BEFORE:"; print self.neural_element[i].value
#                print "ADD   :"; print delta_pool[i]
#                print "ADD/n :"; print delta_pool[i] / n
                self.neural_element[i].value -= delta_pool[i]
#                print "AFTER :";print self.neural_element[i].value


    # test (using input signals to reach teach signals)
    # input_data : input signals, same dimeision as number of input layer's node.
    # teach_data : teach signals, same dimeision as number of output layer's node.
    def test(self, input_data, teach_data):
        # set input and teach signals, and caliculate output and err signals.
        self.input_signals(input_data); self.teach_signals(teach_data); self.output_signals(); self.error_signals()
    
                
    def add_node(self, add_network_dims, output_funcion = 'sigmoid'):
        # make number of element & layer
        self.number_of_layer += len(add_network_dims);
        self.number_of_element += len(add_network_dims)*2-1;
        # make nodes and weights of all layers
        func = ['linear'] + [output_funcion for i in range(0, len(network_dims)-1)]
        for i in range(0, self.number_of_layer): self.neural_element.append(NeuroNode().make_elment([network_dims[i]], func[i]))
        for i in range(0, self.number_of_layer-1): self.neural_element.insert(i*2+1, NeuroWeight().make_elment(np.array(network_dims[i:i+2])))
        # elements_names presents that a element of parameter of 'neural_elements' is 'node', 'weight', 'error'.
        for i in range(0, self.number_of_layer): self.name_of_element.append('node'); self.name_of_element.append('weight')
        self.name_of_element.pop()
        # make teach & error signals
        self.teach = np.matrix([np.zeros(network_dims[len(network_dims)-1])]); self.error = np.matrix([np.zeros(network_dims[len(network_dims)-1])])
        return self
    
    def get_max_output_index(self):
        return np.argmax(self.neural_element[len(self.neural_element)-1].value)

    def get_min_output_index(self):
        return np.argmin(self.neural_element[len(self.neural_element)-1].value)

    def get_output(self):
        return self.neural_element[len(self.neural_element)-1].value

    def get_error(self):
        return self.error

