import cv2

from enhancement import * 
from segmentation import *

import numpy as np
from matplotlib import pyplot as plt

###########################
#         LOADING         #
###########################
# cv2.namedWindow("Original", cv2.WINDOW_NORMAL)    # Create window with freedom of dimensions
im = cv2.imread("sample images/AND_10.jpg")                    # Read image
img = cv2.resize(im, (540, 540))                # Resize image
cv2.imshow("output", img)   

# img=cv2.imread("sample images/image3.jpg",cv2.IMREAD_COLOR)
# original = img.copy()
# cv2.imshow("Original", img)

enhancedImg = enhance(img)
cv2.imshow("enhanced", enhancedImg)

# segmentedImg = segment(original, enhancedImg)
# cv2.imshow("segmented", segmentedImg)


# img_float32 = np.float32(enhancedImg)

# dft = cv2.dft(img_float32, flags = cv2.DFT_COMPLEX_OUTPUT)
# dft_shift = np.fft.fftshift(dft)

# magnitude_spectrum = 20*np.log(cv2.magnitude(dft_shift[:,:,0],dft_shift[:,:,1]))

# plt.subplot(121),plt.imshow(img, cmap = 'gray')
# plt.title('Input Image'), plt.xticks([]), plt.yticks([])
# plt.subplot(122),plt.imshow(magnitude_spectrum, cmap = 'gray')
# plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
# plt.show()

# print(dft_shift)


# print(magnitude_spectrum)

cv2.waitKey(0)