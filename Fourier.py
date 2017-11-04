#!/usr/bin/env python
import sys
import cv2
import numpy as np
from matplotlib import pyplot as plt
import argparse
import math

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
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret,img_thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
# cv2.imshow("origin", img)

rows,cols = img.shape[:2]
nrows = cv2.getOptimalDFTSize(rows)
ncols = cv2.getOptimalDFTSize(cols)
print nrows, ncols

# nimg=np.zeros((nrows,ncols))
# nimg[:rows,:cols]=img
 
right = ncols - cols
bottom = nrows - rows
# just to avoid line breakup in PDF file
bordertype = cv2.BORDER_CONSTANT
nimg = cv2.copyMakeBorder(img_thresh,0,bottom,0,right,bordertype,value=0)
#print nimg.shape


dft = cv2.dft(np.float32(nimg),flags = cv2.DFT_COMPLEX_OUTPUT)
dft_shift = np.fft.fftshift(dft)

magnitude_spectrum = 20*np.log(cv2.magnitude(dft_shift[:,:,0],dft_shift[:,:,1]))
# log_mat = magnitude_spectrum
log_mat= 20*cv2.log(cv2.magnitude(dft_shift[:,:,0], dft_shift[:,:,1]))
cv2.normalize(log_mat, log_mat, 0, 255, cv2.NORM_MINMAX)
img_spectrum = np.uint8(np.around(log_mat))

bgr_spectrum = cv2.cvtColor(img_spectrum, cv2.COLOR_GRAY2BGR)

ret2,img_thresh2 = cv2.threshold(img_spectrum, 150, 255, cv2.THRESH_BINARY)
edges = cv2.Canny(img_thresh2 ,100,200,apertureSize = 3)
lines = cv2.HoughLines(edges,1,np.pi/180, 150)
if lines is None :
    lines = cv2.HoughLines(edges,1,np.pi/180, 140)
if lines is None :
    lines = cv2.HoughLines(edges,1,np.pi/180, 100)
if lines is None  :
    print "Error: don\'t find target"
    cv2.imshow("origin_spectrum", img_spectrum)
    cv2.imshow("thresh_spectrum", img_thresh2)
    keycode = cv2.waitKey()
    cv2.destroyAllWindows()
    sys.exit(1)

angles = []
for line in lines:
    rho,theta = line[0]
    ang = math.degrees(theta)
    print "angle =", ang
    angles.append(int(ang))
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a*rho
    y0 = b*rho
    x1 = int(x0 + 1000*(-b))
    y1 = int(y0 + 1000*(a))
    x2 = int(x0 - 1000*(-b))
    y2 = int(y0 - 1000*(a))

    #cv2.line(img_spectrum,(x1,y1),(x2,y2),(0,0,255),3)
    cv2.line(bgr_spectrum,(x1,y1),(x2,y2),(0,255,0),2)

# cv2.imshow("origin_spectrum", img_spectrum)
# cv2.imshow("thresh_spectrum", img_thresh2)
keycode = cv2.waitKey()
cv2.destroyAllWindows()
if len(angles) > 0 :
    found = False
    angles = list(set(angles))
    for angle in angles :
        print "angle", angle
        if angle != 90 and angle != 0 :
            found = True;
    print "found =", found
    print "angles =", angles
    if found :
        for angle in angles:
            if angle == 90 :
                angles.remove(90)
        for angle in angles:
            if angle == 0 :
                angles.remove(0)
    print "After remove, angles =", angles
    angles = sorted(angles)
    angle = angles[-1]
    angle = 90 - angle
else :
    angle = 90

dst = rotate_bound(img, angle)

plt.subplot(221),plt.imshow(img, cmap = 'gray')
plt.title('Input Image'), plt.xticks([]), plt.yticks([])
plt.subplot(223),plt.imshow(bgr_spectrum, cmap = 'gray')
plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])

plt.subplot(222),plt.imshow(img_thresh2, cmap = 'gray')
plt.title('Threshod Spectrum'), plt.xticks([]), plt.yticks([])
plt.subplot(224),plt.imshow(dst, cmap = 'gray')
plt.title('Rotated Image'), plt.xticks([]), plt.yticks([])
plt.show()

plt.subplot(111),plt.imshow(dst, cmap = 'gray')
plt.title('rotaed img'), plt.xticks([]), plt.yticks([])
plt.show()
# cv2.imshow("rotaed img", dst)
# keycode = cv2.waitKey()
# cv2.destroyAllWindows()
