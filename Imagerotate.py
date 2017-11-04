#!/usr/bin/env python
#-*-coding:utf-8-*-
import cv2
import numpy as np
import math
import argparse

def rotate_bound(image, angle):
    # grab the dimensions of the image and then determine the
    # center
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)
 
    # grab the rotation matrix (applying the negative of the
    # angle to rotate clockwise), then grab the sine and cosine
    # (i.e., the rotation components of the matrix)
    M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
 
    # compute the new bounding dimensions of the image
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))
 
    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY
 
    # perform the actual rotation and return the image
    return cv2.warpAffine(image, M, (nW, nH), borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))

ap = argparse.ArgumentParser()
ap.add_argument("-i","--image", required = True, help = "Path to image file")
ap.add_argument("-d","--degrees", required = False, help = "Degrees for rotated")
args = vars(ap.parse_args())

img_file = args["image"]
tmp_degrees = args["degrees"]

fltDegrees = 0
if None != tmp_degrees :
    fltDegrees = float(tmp_degrees)
print fltDegrees

img = cv2.imread(img_file)
# rows,cols = img.shape[:2]
# M = cv2.getRotationMatrix2D((rows/2,cols/2), fltDegrees,1)
# dst = cv2.warpAffine(img,M,(2*rows, 2*rows), borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))

dst = rotate_bound(img, fltDegrees)

cv2.imshow("origin", img)
cv2.imshow("rotated", dst)
cv2.waitKey()
cv2.destroyAllWindows()
