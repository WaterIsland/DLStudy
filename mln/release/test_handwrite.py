#!/usr/bin/python
# -*- coding: utf-8 -*-
#
# This precision method refered by
#   http://scikit-learn.org/stable/auto_examples/classification/plot_digits_classification.html
#   https://github.com/sylvan5/PRML/blob/master/ch5/digits.py
#


import cv2
import numpy as np
import Mln as mln
import dump as dp
import progress as prg
import image as img

from sklearn.datasets import load_digits
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import confusion_matrix, classification_report



def recognition_digit_image(fname, digit = 100):
    im = cv2.imread(fname)
    im = img.change_size_with_size(im, 28, 28)
    im = img.change_grayscale(im)
    im = 255. - im
    input_data = im
    input_data = input_data.astype(np.float64)
    input_data = im / im.max()
    tmp_data = np.reshape(input_data, (28*28, 1))

    neuro_obj.test(tmp_data, teach_data)
    output = neuro_obj.get_output()

    if digit >=0 and digit <= 9:
        if neuro_obj.get_max_output_index() == digit : print "judged(success):", neuro_obj.get_max_output_index()
        else                                         : print "judged(miss)   :", neuro_obj.get_max_output_index()
    else:
        print "judged:", neuro_obj.get_max_output_index()
    
    # if you wanna show image then you delete comment-out fllowing 3 lines.
    im = cv2.resize(im, (28*10, 28*10))
    cv2.imshow("input_data", im)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


teach_data = [0, 0, 0, 1, 0, 0, 0, 0, 0, 0] # dummy

print '--start--'
print '@@ Show after learning @@'

neuro_obj = dp.obj_load('./learn-classification.pkl') # online training's pkl
#neuro_obj = dp.obj_load('./learn-classification-batch.pkl') # batch training's pkl

recognition_digit_image("../../hand-writing-number/0.png", 0)
recognition_digit_image("../../hand-writing-number/1.png", 1)
recognition_digit_image("../../hand-writing-number/2.png", 2)
recognition_digit_image("../../hand-writing-number/3.png", 3)
recognition_digit_image("../../hand-writing-number/4.png", 4)
recognition_digit_image("../../hand-writing-number/5.png", 5)
recognition_digit_image("../../hand-writing-number/6.png", 6)
recognition_digit_image("../../hand-writing-number/7.png", 7)
recognition_digit_image("../../hand-writing-number/8.png", 8)
recognition_digit_image("../../hand-writing-number/9.png", 9)


# judging '9' is success.
recognition_digit_image("../../hand-writing-number/number.png")
