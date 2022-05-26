import cv2
import numpy as np
from scipy import ndimage

###########################
#        DENOISING        #
###########################
def denoising(img):
    return cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 15) 

###########################
#      BINARIZATION       #
###########################
# convert the image to grayscale and blur it slightly
def binarize(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    # using adaptive threshold
    return cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 21, 10)

###########################
#     SKEW CORRECTION     #
###########################
def find_score(arr, angle):
    data = ndimage.rotate(arr, angle, reshape=False, order=0)
    hist = np.sum(data, axis=1)
    score = np.sum((hist[1:] - hist[:-1]) ** 2)
    return hist, score

def skewCorrection(img):
    delta = 1
    limit = 5
    angles = np.arange(-limit, limit+delta, delta)
    scores = []
    for angle in angles:
        hist, score = find_score(img, angle)
        scores.append(score)
    best_score = max(scores)
    best_angle = angles[scores.index(best_score)]

    # correct skew
    return ndimage.rotate(img, best_angle, reshape=False, order=0)

###########################
#        DILATION         #
###########################
def dilate(img):
    kernel = np.ones((3,3), np.uint8)
    return cv2.dilate(img, kernel, iterations = 2)

###########################
#        EROSION         #
###########################
def erode(img):
    kernel = np.ones((3,3), np.uint8)
    return cv2.erode(img, kernel, iterations = 1)


def enhance(img):
    denoisedImage = denoising(img)
    binarizedImage = binarize(denoisedImage)
    #skewCorrectedImage = skewCorrection(binarizedImage)
    dilatedImage = dilate(binarizedImage)
    erodedImage = erode(dilatedImage)
    return erodedImage
