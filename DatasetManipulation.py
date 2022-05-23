import cv2
import os
from enhancement import * 
from segmentation import *
from skimage.io import imread
from skimage.transform import resize
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd

################################
#   Generating Features and    #
#   store in a CSV File.       #
################################
Categories=['AND','OR','NOT']
flat_data_arr=[]
target_arr=[]
datasetPath = 'dataset/Training/Enhanced'

for i in Categories:
    print(f'loading... category : {i}')
    path=os.path.join(datasetPath,i)
    for imgName in os.listdir(path):
        image = cv2.imread(path + '/' + imgName)
        sift = cv2.SIFT_create()
        kp, des = sift.detectAndCompute(image, None)
        des_resized=resize(des,(50,50,1))
        flat_data_arr.append(des_resized.flatten())
        target_arr.append(Categories.index(i))
    print(f'loaded category:{i} successfully')

flat_data=np.array(flat_data_arr)
target=np.array(target_arr)
df=pd.DataFrame(flat_data) #dataframe
df['Target']=target
df.to_csv('SIFT Features Testing.csv')


################################
#   Loop to generate enhanced  #
#   dataset.                   #
################################
# datasetPath = 'dataset/Testing/Original'
# Categories=['AND','OR','NOT']
# count = 1
# for i in Categories:
#     print(f'Generating... category : {i}')
#     path=os.path.join(datasetPath,i)
#     for img in os.listdir(path):
#         im=cv2.imread(path + '/' + img)
#         imgResized = cv2.resize(im, (540, 540))
#         enhancedImg = enhance(imgResized)
#         cv2.imwrite(i + str(count)+'.jpg', enhancedImg)
#         count+=1


