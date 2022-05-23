import cv2
import os
from enhancement import * 
from segmentation import *
from skimage.io import imread
from skimage.transform import resize
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd



image = cv2.imread("circuits/image3.jpg")   
enhancedImage = enhance(image)
segmented = segment(image, enhancedImage)





cv2.waitKey(0)