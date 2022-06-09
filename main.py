import cv2
from enhancement import * 
from segmentation import *

image = cv2.imread("samples/circuit3.jpg") 
enhancedImage = enhance(image)
segmented = segment(image, enhancedImage)

cv2.namedWindow("output", cv2.WINDOW_NORMAL)
cv2.imshow("output", segmented)

cv2.waitKey(0)