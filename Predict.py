import pickle
import cv2
from skimage.transform import resize
from skimage.io import imread
from enhancement import * 

def Predict(imgPath):
    ###########################
    #      Model Loading &    #
    #      Image features     #
    ###########################
    Categories=['AND','OR','NOT']
    model=pickle.load(open('models/model80.p','rb'))

    # Read Image
    image = imread(imgPath)

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
    return Categories[model.predict(img_data)[0]]


