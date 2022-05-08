import cv2
import numpy as np


original = image.copy()
thresh = data

# Find lines
minLineLength = 10
maxLineGap = 80
lines = cv2.HoughLinesP(thresh,1,np.pi/180,100,minLineLength,maxLineGap)
for line in lines:
    for x1,y1,x2,y2 in line:
        cv2.line(thresh,(x1,y1),(x2,y2),(0,0,0),5)

# Morphological operations to clean image
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9,9))
close  = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=3)
cnts = cv2.findContours(close, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if len(cnts) == 2 else cnts[1]

# Contour filtering and ROI extraction
ROI_number = 0
for c in cnts:
    area = cv2.contourArea(c)
    if area > 3000:
        x,y,w,h = cv2.boundingRect(c)
        ROI = original[y:y+h,x:x+w]
        cv2.imwrite('ROI_{}.png'.format(ROI_number), ROI)
        cv2.rectangle(image, (x, y), (x+w, y+h), (36, 255, 12), 8)
        ROI_number += 1

cv2.imshow("",thresh)
cv2.imshow("",close)
cv2.imshow("",image)