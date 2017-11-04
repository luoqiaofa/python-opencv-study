#!/usr/bin/env python
import sys
import os
import os.path
import math
import cv2
import numpy as np
import argparse
import zbar
from PIL import Image

DEBUG = False

def opencv_image2pil(cvimg) :
    channles = len(cvimg.shape)
    gray = cvimg
    if channles >= 3 :
        gray = cv2.cvtColor(cvimg, cv2.COLOR_BGR2GRAY)
    pil = Image.fromarray(gray)
    return pil

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


# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required = True, help = "path to the image file")
args = vars(ap.parse_args())

# load the image and convert it to grayscale
img_file = args["image"]

img = cv2.imread(img_file)
# cv2.imshow('image',img)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret,img_thresh = cv2.threshold(gray, 140, 255, cv2.THRESH_BINARY_INV)
if DEBUG :
    cv2.imshow("thresh", img_thresh)
kernel = np.ones((7,7),np.uint8)
dilation = cv2.dilate(img_thresh,kernel,iterations=3)
if DEBUG :
    cv2.imshow('dilation',dilation)

contour_idx = 0
image, contours, hier = cv2.findContours(dilation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
areas = []

scanner = zbar.ImageScanner()
# configure the reader
scanner.parse_config('enable')
for c in contours:
    # get the bounding rect
    x, y, w, h = cv2.boundingRect(c)
    # draw a green rectangle to visualize the bounding rect
    # cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
    area = int(cv2.contourArea(c))
    
    if (area > 10000) and ((h > 70) and (w > 70)):
        areas.append(area)
        # print area, x,y,w,h

        img_roi = img[y:y+h,x:x+w]
        # print img_roi.shape
        tmpname = "img" + str(contour_idx) + ".png"
        # print "tmpname :", tmpname
        # cv2.imwrite(tmpname, img_roi)
        # print "contour_idx =", contour_idx

        # get the min area rect
        rect = cv2.minAreaRect(c)
        box = cv2.boxPoints(rect)
        # convert all coordinates floating point values to int
        box = np.int0(box)
        #print type(box)
        # print box
        # draw a red 'nghien' rectangle
        # cv2.drawContours(img, [box], 0, (0, 255, 0), 2)
        # print box
        x1 = box[0][0]
        y1 = box[0][1]
        x2 = box[1][0]
        y2 = box[1][1]
        x3 = box[3][0]
        y3 = box[3][1]
        d1 = math.sqrt((x2-x1) * (x2-x1) + (y2-y1) * (y2-y1))
        d2 = math.sqrt((x3-x1) * (x3-x1) + (y3-y1) * (y3-y1))
        int_d1 = int(d1)
        int_d2 = int(d2)
        dst_x1 = x1
        dst_x2 = x2
        dst_y1 = y1
        dst_y2 = y2
        dst_d = d1
        if (int_d1 >= int_d2) :
            # cv2.line(img, (x1,y1),(x2,y2),(255,0,0),5)
            pass
        else :
            dst_x2 = x3
            dst_y2 = y3
            dst_d = d2
            # cv2.line(img, (x1,y1),(x3,y3),(255,0,0),5)
        radius = math.sqrt((dst_y2 - dst_y1)*(dst_y2 - dst_y1)) / \
                 math.sqrt(dst_d * dst_d)
        radius = math.asin(radius)
        angle = math.degrees(radius)
        # print "angle =",angle,"(",dst_x1,dst_y1,")","(",dst_x2,dst_y2,")"
        rotated_img = img_roi
        if angle != 0 :
            if ((dst_x1 < dst_x2) and (dst_y1 < dst_y2)) or \
                ((dst_x1 > dst_x2) and (dst_y1 > dst_y2)) :
                    rotated_img = rotate_bound(img_roi, -angle)
            elif ((dst_x1 > dst_x2) and (dst_y1 < dst_y2)) or \
                ((dst_x1 < dst_x2) and (dst_y1 > dst_y2)) :
                    rotated_img = rotate_bound(img_roi, angle)
            else :
                rotated_img = rotate_bound(img_roi, angle)

        if DEBUG :
            fw_name = str(contour_idx) + "_d" + str(angle) + ".png"
            cv2.imwrite(fw_name, rotated_img)
            # font = cv2.FONT_HERSHEY_SIMPLEX
            # cv2.putText(img,tmpname,(x+10,y+10), font, 1,(0,0,255),2,cv2.LINE_AA)
        pil = opencv_image2pil(rotated_img)
        width, height = pil.size
        raw = pil.tobytes()
        # wrap image data
        image = zbar.Image(width, height, 'Y800', raw)
        # scan the image for barcodes
        scanner.scan(image)
        # extract results
        for symbol in image:
            # do something useful with results
            print tmpname, ': decoded', symbol.type, 'symbol', '"%s"' % symbol.data

        contour_idx = contour_idx + 1




# cv2.drawContours(img, [box], 0, (0, 255, 0), 2)
#cv2.rectangle(img,(248, 375),(248, 195),(0,255,0),3)

if DEBUG :
    print "areas =", sorted(areas)
    cv2.imshow('image',img)
    cv2.waitKey()
    cv2.destroyAllWindows()

