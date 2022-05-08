import cv2
from preprocessing import enhancement

###########################
#         LOADING         #
###########################
img=cv2.imread("sample images/image5.jpg",cv2.IMREAD_COLOR)
cv2.imshow("Original", img)

cv2.waitKey(0)