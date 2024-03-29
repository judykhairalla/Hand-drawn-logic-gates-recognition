import pandas as pd
import os
import pickle
from skimage.transform import resize
from skimage.io import imread
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,accuracy_score,confusion_matrix

###########################
#     Loading Dataset     #
#   &    Dataframe        #
###########################
'''
Categories=['AND','OR','NOT']

flat_data_arr=[]
target_arr=[]

datasetPath = 'dataset/Training'

for i in Categories:
    print(f'loading... category : {i}')
    path=os.path.join(datasetPath,i)
    for img in os.listdir(path):
        img_array=imread(os.path.join(path,img))
        img_resized=resize(img_array,(150,150,3))
        flat_data_arr.append(img_resized.flatten())
        target_arr.append(Categories.index(i))
    print(f'loaded category:{i} successfully')

flat_data=np.array(flat_data_arr)
target=np.array(target_arr)
df=pd.DataFrame(flat_data) #dataframe
df['Target']=target

df.to_csv('img_data.csv')
'''
df = pd.read_csv('CSVFiles/SIFT Features Training128.csv')
x_train=df.iloc[:,1:-1] #input data 
y_train=df.iloc[:,-1]



df = pd.read_csv('CSVFiles/SIFT Features Testing128.csv')
x_test=df.iloc[:,1:-1] #input data 
y_test=df.iloc[:,-1]


###########################
#      Model Creation     #
###########################
param_grid={'C':[0.1,1,10],'gamma':[0.001,0.1,1],'kernel':['rbf','poly']}
svc=svm.SVC(probability=True)
model=GridSearchCV(svc,param_grid)


###########################
#      Model Training     #
###########################
model.fit(x_train,y_train)
print('The Model is trained with the given images')
pickle.dump(model,open('model128.p','wb'))
print("Pickle is dumped successfully")


###########################
#      Model Testing      #
###########################
y_pred=model.predict(x_test)
print("The predicted Data is :")
print(y_pred)
print("The actual data is:")
print(np.array(y_test))
print(f"The model is {accuracy_score(y_pred,y_test)*100}% accurate")