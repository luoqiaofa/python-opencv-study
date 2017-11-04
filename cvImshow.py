#!/usr/bin/env python
# -*- coding: UTF-8 -*-
import sys
import os
import os.path
import cv2
import numpy as np
import argparse

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required = True, help = "path to the image file")
args = vars(ap.parse_args())

# load the image and convert it to grayscale
img_file = args["image"]
if os.path.exists(img_file) is False :
    print "File: \'%s\' don\'t exist" % img_file
    sys.exit(-1)

img = cv2.imread(img_file)
height,width = img.shape[:2]
print height,width

base_fname = os.path.basename(img_file)

cv2.imshow(base_fname, img)

cv2.waitKey()
cv2.destroyAllWindows()

