#!/usr/bin/python
# -*- coding: UTF-8 -*-
# https://raw.githubusercontent.com/tarikd/barcode_detection/master/detect_barcode.py
# USAGE
# python detect_barcode.py --image images/barcode_01.jpg

# import the necessary packages
import numpy as np
import argparse
import cv2
import os
import os.path

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required = True, help = "path to the image file")
args = vars(ap.parse_args())

# load the image and convert it to grayscale
image = cv2.imread(args["image"])


gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

base_file_name = os.path.splitext(args["image"])[0]

# gray_file_name = base_file_name + "_gray.jpg"
# cv2.imwrite(gray_file_name,gray)

# compute the Scharr gradient magnitude representation of the images
# in both the x and y direction
gradX = cv2.Sobel(gray, ddepth = cv2.CV_32F, dx = 1, dy = 0, ksize = -1)
gradY = cv2.Sobel(gray, ddepth = cv2.CV_32F, dx = 0, dy = 1, ksize = -1)

# subtract the y-gradient from the x-gradient
gradient = cv2.subtract(gradX, gradY)
gradient = cv2.convertScaleAbs(gradient)

# blur and threshold the image
blurred = cv2.blur(gradient, (9, 9))
(retValue, thresh) = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY_INV)


# cv2.imshow("Image", blr)
# blr_media = cv2.medianBlur(image, 5)
(retValue, dst) = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)
# thresh_file_name = base_file_name + "_thresh.jpg"
# cv2.imwrite(thresh_file_name, dst)

# construct a closing kernel and apply it to the thresholded image
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 21))
closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

# perform a series of erosions and dilations
closed = cv2.erode(closed, None, iterations = 4)
closed = cv2.dilate(closed, None, iterations = 4)

# find the contours in the thresholded image, then sort the contours
# by their area, keeping only the largest one
img, cnts, heir = cv2.findContours(closed.copy(), cv2.RETR_EXTERNAL,
	cv2.CHAIN_APPROX_SIMPLE)
print "num:", len(cnts)

# c = sorted(cnts, key = cv2.contourArea, reverse = True)[0]
# rect = cv2.minAreaRect(c)
# box = np.int0(cv2.boxPoints(rect))
# print box
# cv2.drawContours(image, [box], -1, (0, 255, 0), 2)
for c in cnts :
    # compute the rotated bounding box of the largest contour
    rect = cv2.minAreaRect(c)
    box = np.int0(cv2.boxPoints(rect))
    print box
    x1 = box[0][0]
    x2 = box[2][0]
    y1 = box[1][1]
    y2 = box[3][1]
    win_name = "rect0"
    print x1,x2,y1,y2
    # print img_rect
    
    # draw a bounding box arounded the detected barcode and display the
    # image
    cv2.drawContours(image, [box], -1, (0, 255, 0), 2)

cv2.imshow("barcode", image)

cv2.waitKey(0)
cv2.destroyAllWindows()

