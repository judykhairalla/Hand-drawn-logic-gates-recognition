import cv2
import numpy as np
import os

from predict import Predict
# Find lines
def removeLines(img):
    minLineLength = 10
    maxLineGap = 80
    lines = cv2.HoughLinesP(img,1,np.pi/180,100,minLineLength,maxLineGap)
    for line in lines:
        for x1,y1,x2,y2 in line:
            cv2.line(img,(x1,y1),(x2,y2),(0,0,0),5)
    return img

# Morphological operations to clean image
def contourDetection(img):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9,9))
    close  = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel, iterations=3)
    contours = cv2.findContours(close, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours[0] if len(contours) == 2 else contours[1]

# Contour filtering and ROI extraction
def drawBoundingBoxes(origImg, contours):
    ROI_number = 0
    path = 'ROIs'
    
    isExist = os.path.exists(path)
    if not isExist:
        os.makedirs(path)
    
    for c in contours:
        area = cv2.contourArea(c)
        if area > 3000:
            x,y,w,h = cv2.boundingRect(c)
            ROI = origImg[y:y+h,x:x+w]
            imagePath = os.path.join(path, 'ROI_{}.png'.format(ROI_number))
            cv2.imwrite(imagePath, ROI)
            label = Predict(imagePath)
            cv2.rectangle(origImg, (x, y), (x+w, y+h), (36, 255, 12), 8)
            cv2.putText(img=origImg, text=label, org=(x, y+h+60), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=2, color=(255, 255, 255),thickness=4)
            ROI_number += 1
    return origImg

def segment(origImg, enhancedImg):
    removedLinesImg = removeLines(enhancedImg)
    contours = contourDetection(removedLinesImg)
    boundingBoxesImg = drawBoundingBoxes(origImg, contours)
    return boundingBoxesImg
