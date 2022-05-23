import pickle
from skimage.transform import resize
from skimage.io import imread
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,accuracy_score,confusion_matrix
import cv2
from enhancement import * 

###########################
#      Model Loading &    #
#      Image features     #
###########################
model=pickle.load(open('img_model.p','rb'))

# Read Image
img_path = "ROI_0.png"
image = imread(img_path)
imgResized = cv2.resize(image, (540, 540))

# Enhance image
enhancedImg = enhance(imgResized)

# Extract Features
sift = cv2.SIFT_create()
kp, des = sift.detectAndCompute(enhancedImg, None)
des_resized=resize(des,(50,50,1))
img_data=[des_resized.flatten()]

# Predict
probability=model.predict_proba(img_data)



###########################
#         RESULTS         #
###########################
Categories=['AND','OR','NOT']
for ind,val in enumerate(Categories):
  print(f'{val} = {probability[0][ind]*100}%')
print("The predicted image is : "+Categories[model.predict(img_data)[0]])



imgCV=cv2.imread(img_path,cv2.IMREAD_COLOR)
imS = cv2.resize(imgCV, (500, 500))
cv2.imshow("Original", imS)
cv2.waitKey(0) 
