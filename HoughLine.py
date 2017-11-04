#!/usr/bin/env python
#-*-coding:utf-8-*-
import cv2
import numpy as np
import math
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-i","--image", required = True, help = "Path to image file")
args = vars(ap.parse_args())

img_file = args["image"]

img = cv2.imread(img_file)
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
ret,thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
cv2.imshow("threshold", thresh)

edges = cv2.Canny(thresh,100,200,apertureSize = 3)
lines = cv2.HoughLines(edges,1,np.pi/180, 500)
if lines is None :
    pass
else :
    print len(lines)
    for line in lines:
        rho,theta = line[0]
        # print math.degrees(theta)
        print math.degrees(theta)
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a*rho
        y0 = b*rho
        x1 = int(x0 + 1000*(-b))
        y1 = int(y0 + 1000*(a))
        x2 = int(x0 - 1000*(-b))
        y2 = int(y0 - 1000*(a))
        # print "[",x1,",",y1,"]", "[",x2,",",y2,"]"
        cv2.line(img,(x1,y1),(x2,y2),(0,0,255),2)

cv2.imshow('houghlines',img)

cv2.waitKey()
cv2.destroyAllWindows()
