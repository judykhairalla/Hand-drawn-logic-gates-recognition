import pickle
import cv2
from skimage.transform import resize
from skimage.io import imread
from enhancement import * 

###########################
#      Model Loading &    #
#      Image features     #
###########################
model=pickle.load(open('models/model80.p','rb'))

# Read Image
img_path = "ROIs/ROI_1.png"
image = imread(img_path)

# Enhance image
enhancedImg = enhance(image)

# Extract Features
sift = cv2.SIFT_create()
kp, des = sift.detectAndCompute(enhancedImg, None)
des_resized=resize(des,(80,80,1))
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
cv2.imshow("Original", enhancedImg)
cv2.waitKey(0) 
