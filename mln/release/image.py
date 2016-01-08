#!/usr/bin/python

import cv2
import numpy as np

# Reffered by http://tatabox.hatenablog.com/entry/2013/07/15/164015
def change_size_with_rate(image, width_rate, height_rate):
	hight = int(image.shape[0] * width_rate)
	width = int(image.shape[1] * height_rate)
	change_image = cv2.resize(image, (hight, width))

#	cv2.imshow("change_image", change_image)
#	cv2.waitKey(0)
#	cv2.destroyAllWindows()  
	return change_image

# Reffered by http://tatabox.hatenablog.com/entry/2013/07/15/164015
def change_size_with_size(image, width, height):
	hight = int(width)
	width = int(height)
	change_image = cv2.resize(image, (hight, width))

#	cv2.imshow("change_image", change_image)
#	cv2.waitKey(0)
#	cv2.destroyAllWindows()  
	return change_image

# Reffered by https://github.com/tanaka0079/python/blob/python/opencv/image/grayscale.py
def change_grayscale(image):
	change_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

#	cv2.imshow("change_image", change_image)
#	cv2.waitKey(0)
#	cv2.destroyAllWindows()  
	return change_image

if __name__ == '__main__':
    im = cv2.imread("image/number.png")
    im2 = change_size_with_rate(im, 2, 2)
    im3 = change_size_with_rate(im, 4, 4)
    im4 = change_size_with_size(im, 50, 50)
    im5 = change_size_with_size(im, 25, 25)
    im31 = change_grayscale(im)

    print im.shape
    print im2.shape
    print im3.shape
    print im4.shape
    print im5.shape
    print im31.shape

#    print im31

#    if not (im == None):
#    if not (im):
#        half_size(im)
#    else:
#        print 'Not exist'

