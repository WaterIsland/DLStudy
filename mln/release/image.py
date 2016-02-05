#!/usr/bin/python

import cv2
import numpy as np

# Reffered by http://tatabox.hatenablog.com/entry/2013/07/15/164015
def change_size_with_rate(image, width_rate, height_rate):
	hight = int(image.shape[0] * width_rate)
	width = int(image.shape[1] * height_rate)
	change_image = cv2.resize(image, (hight, width))

	return change_image

# Reffered by http://tatabox.hatenablog.com/entry/2013/07/15/164015
def change_size_with_size(image, width, height):
	hight = int(width)
	width = int(height)
	change_image = cv2.resize(image, (hight, width))

	return change_image

# Reffered by https://github.com/tanaka0079/python/blob/python/opencv/image/grayscale.py
def change_grayscale(image):
	change_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

	return change_image
