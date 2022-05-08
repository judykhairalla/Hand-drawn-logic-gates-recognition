import cv2
import numpy as np
from PIL import Image as im
from scipy import ndimage

###########################
#        DENOISING        #
###########################
dst = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 15) 
# cv2.imshow("Noise removed",dst)

###########################
#      BINARIZATION       #
###########################
# convert the image to grayscale and blur it slightly
gray = cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (7, 7), 0)

# using adaptive threshold
thresh = cv2.adaptiveThreshold(blurred, 255,
	cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 21, 10)
# cv2.imshow("Binarized", thresh)

###########################
#     SKEW CORRECTION     #
###########################
def find_score(arr, angle):
    data = ndimage.rotate(arr, angle, reshape=False, order=0)
    hist = np.sum(data, axis=1)
    score = np.sum((hist[1:] - hist[:-1]) ** 2)
    return hist, score

delta = 1
limit = 5
angles = np.arange(-limit, limit+delta, delta)
scores = []
for angle in angles:
    hist, score = find_score(thresh, angle)
    scores.append(score)
best_score = max(scores)
best_angle = angles[scores.index(best_score)]

# correct skew
data = ndimage.rotate(thresh, best_angle, reshape=False, order=0)
img = im.fromarray((255 * data).astype("uint8")).convert("RGB")
# cv2.imshow("Skew correction",data)

###########################
#        DILATION         #
###########################
kernel = np.ones((3,3), np.uint8)
data = cv2.dilate(data, kernel, iterations = 1)
cv2.imshow("Dilation",data)

cv2.waitKey(0)