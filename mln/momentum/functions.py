#!/usr/bin/python
# coding=utf-8

# show mnist image data on CUI
def show_image_and_label(image, label, n):
    xxx = image[n]
    yyy = label[n]

    print ""
    print "Label:"; print yyy
    print "Image:"
    for i in range(0, 28):
        print ""
        for j in range(0, 28):
            if xxx[i*28 + j] > 0.: print "#",
            else                 : print " ",

