#!/usr/bin/env python
# -*- coding: UTF-8 -*-
import cv2
import numpy as np
import argparse

THRESHOLD_COLOR_H = 110
THRESHOLD_COLOR_L = 11

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required = True, help = "path to the image file")
args = vars(ap.parse_args())

# load the image and convert it to grayscale
img_file = args["image"]

img = cv2.imread(img_file)
height,width = img.shape[:2]
print height,width

cv2.imshow("origin", img)

for h in range(height) :
    for w in range(width) :
        b = img.item(h,w,0)
        g = img.item(h,w,1)
        r = img.item(h,w,2)
        # if r > 250 and g < 230 and b < 230 :
        #     img[h,w] = [255,255,255]
        # if r > 200 and g < 200 and b < 200 :
        #     img[h,w] = [255,255,255]
        # if r > 160 and g < 150 and b < 150 :
        #     img[h,w] = [255,255,255]
        # if r > 100 and g < 10 and b < 10 :
        #     img[h,w] = [255,255,255]
        # if r > 150 and g < 15 and b < 15 :
        #     img[h,w] = [255,255,255]
        # if r > 130 and g < 20 and b < 20 :
        #     img[h,w] = [255,255,255]

        if r > 230 and g < 160 and b < 160 :
            img[h,w] = [255,255,255]
        if r > 200 and g < 150 and b < 150 :
            img[h,w] = [255,255,255]
        if r > 150 and g < 20 and b < 20 :
            img[h,w] = [255,255,255]
        if r > 200 and g > 200 and b > 230 :
            img[h,w] = [255,255,255]
        if r < 10 and g < 10 and b < 10 :
            img[h,w] = [255,255,255]
        if r > 230 :
            if g < 170 or b < 170 :
                img[h,w] = [255,255,255]
            if g < 200 or b < 200 :
                img[h,w] = [255,255,255]
        if r > 180 :
            if g < 100 or b < 100 :
                img[h,w] = [255,255,255]
        if r > 160 :
            if g < 100 or b < 100 :
                img[h,w] = [255,255,255]


cv2.imshow("ColorFilter", img)

cv2.waitKey()
cv2.destroyAllWindows()

