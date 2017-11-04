#!/usr/bin/env python
import sys
import os
import os.path
import cv2
import numpy as np

import argparse

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required = True, help = "path to the image file")
ap.add_argument("-t", "--threshold", required = False, help = "threshold to convert")
args = vars(ap.parse_args())

# load the image and convert it to grayscale
img_file = args["image"]
if os.path.exists(img_file)  :
    pass
else :
    print "Error: Image file \"%s\" don't exist!" % img_file
    sys.exit(-1)

img = cv2.imread(img_file, cv2.IMREAD_UNCHANGED)
gray = img
if len(img.shape) == 3 :
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

th_int = 127
th_str = args["threshold"]
if th_str != None :
    th_int = int(th_str)

ret,th_img = cv2.threshold(gray, th_int, 255,cv2.THRESH_BINARY)

cv2.imshow("origin", img)
cv2.imshow("gray", gray)
th_fname = "threshold(%s)" % th_int
cv2.imshow(th_fname, th_img)

key = cv2.waitKey(-1) & 0xff
cv2.destroyAllWindows()

