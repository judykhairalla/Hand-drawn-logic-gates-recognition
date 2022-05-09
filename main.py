import cv2

from enhancement import * 
from segmentation import *

###########################
#         LOADING         #
###########################
img=cv2.imread("sample images/image3.jpg",cv2.IMREAD_COLOR)
original = img.copy()
# cv2.imshow("Original", img)

enhancedImg = enhance(img)
cv2.imshow("enhanced", enhancedImg)

segmentedImg = segment(original, enhancedImg)
cv2.imshow("segmented", segmentedImg)


cv2.waitKey(0)