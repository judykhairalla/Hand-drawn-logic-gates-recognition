import pickle
from skimage.transform import resize
from skimage.io import imread
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,accuracy_score,confusion_matrix
import cv2

Categories=['AND','OR','NOT']
model=pickle.load(open('img_model.p','rb'))

img_path = "ROI_2.png"

img=imread(img_path)

img_resize=resize(img,(150,150,3))
img_data=[img_resize.flatten()]
probability=model.predict_proba(img_data)

for ind,val in enumerate(Categories):
  print(f'{val} = {probability[0][ind]*100}%')
print("The predicted image is : "+Categories[model.predict(img_data)[0]])



imgCV=cv2.imread(img_path,cv2.IMREAD_COLOR)
imS = cv2.resize(imgCV, (500, 500))
cv2.imshow("Original", imS)
cv2.waitKey(0) 
