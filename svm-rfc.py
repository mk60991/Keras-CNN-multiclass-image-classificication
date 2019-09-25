# -*- coding: utf-8 -*-
"""
Created on Sat Sep 21 15:51:55 2019

@author: Manish
"""

import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
%matplotlib inline
import tensorflow as tf
import keras
import glob
import cv2
import pickle, datetime

from sklearn.preprocessing import LabelBinarizer

from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import os


print(os.listdir(r"C:\Users\hp\Desktop\neuroequilibrium-assignment-2\Assign-1-cnn-svm-rfc\data"))



#image preprocessing
#for training

train_fruit_images = []
train_fruit_labels = [] 
for directory_path in glob.glob("data/train/*"):
    fruit_label = directory_path.split("\\")[-1]
    for img_path in glob.glob(os.path.join(directory_path, "*.jpg")):
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)       
        img = cv2.resize(img, (227, 227))
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        train_fruit_images.append(img)
        train_fruit_labels.append(fruit_label)
train_fruit_images = np.array(train_fruit_images)
train_fruit_labels = np.array(train_fruit_labels)


label_to_id = {v:i for i,v in enumerate(np.unique(train_fruit_labels))}
id_to_label = {v: k for k, v in label_to_id.items()}
train_label_ids = np.array([label_to_id[x] for x in train_fruit_labels])



train_fruit_images.shape, train_label_ids.shape, train_fruit_labels.shape



#test

test_fruit_images = []
test_fruit_labels = [] 
for directory_path in glob.glob("data/test/*"):
    fruit_label = directory_path.split("\\")[-1]
    for img_path in glob.glob(os.path.join(directory_path, "*.jpg")):
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        
        img = cv2.resize(img, (227, 227))
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        
        test_fruit_images.append(img)
        test_fruit_labels.append(fruit_label)
test_fruit_images = np.array(test_fruit_images)
test_fruit_labels = np.array(test_fruit_labels)



test_label_ids = np.array([label_to_id[x] for x in test_fruit_labels])

test_fruit_images.shape, test_label_ids.shape



#splitting of data into training and tresting
x_train, y_train, x_test, y_test, N_CATEGORY =train_fruit_images,train_fruit_labels,test_fruit_images,test_fruit_labels,len(label_to_id)

print(x_train.shape, y_train.shape, x_test.shape, y_test.shape, N_CATEGORY)


print(id_to_label)



#X_normalized = np.array(x_train )
#X_normalized_test = np.array(x_test )

#label encoding
labelencoder = LabelEncoder()
y_one_hot = labelencoder.fit_transform(y_train)
y_one_hot_test = labelencoder.fit_transform(y_test)

#reshaping train features
nsamples, nx, ny,nz = x_train.shape
d2_train_dataset = x_train.reshape((nsamples,nx*ny*nz))



#reshaping test features
nsamples1, nx1, ny1,nz1 = x_test.shape
d2_test_dataset = x_test.reshape((nsamples1,nx1*ny1*nz1))

#random forest classifier
#Feed the extracted features with the labels to RANDOM FOREST 
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators = 20, random_state = 42)

rf.fit(d2_train_dataset, y_one_hot)


#Feed the features of the test images to Random Forest Classifier to predict its class
predictions1 = rf.predict(d2_test_dataset)

#accuracy score
accuracy=accuracy_score(predictions1 , y_one_hot_test)
print("accuracy for Random Forest :")
print('Accuracy:', accuracy*100, '%.')






#support vector machine (svm)
from sklearn.svm import SVC

SVM = SVC(kernel='linear', random_state=0)

SVM.fit(d2_train_dataset, y_one_hot)
#Feed the features of the test images to Random Forest Classifier to predict its class
predictions2 = SVM.predict(d2_test_dataset)

#accuracy_score
accuracy=accuracy_score(predictions2 , test_label_ids)
print("accuracy for SVM:")
print('Accuracy:', accuracy*100, '%.')








#prediction on individual image
img_width, img_height = 227,227

# Get test image ready
img_path=r"C:\Users\hp\Desktop\neuroequilibrium-assignment-2\Assign-1-cnn-svm-rfc\data\validation\image_0051.jpg"
test_image = cv2.imread(img_path, cv2.IMREAD_COLOR)     
test_image = cv2.resize(test_image, (img_width, img_height))

test_image = np.expand_dims(test_image, axis=0)

test_image = test_image.reshape(1,img_width, img_height,3)    # Ambiguity!
# Should this instead be: test_image.reshape(img_width, img_height, 3) ??
nsamples, nx, ny,nz = test_image.shape
d2_image = test_image.reshape((nsamples,nx*ny*nz))

result = rf.predict(d2_image)
print (result)


#Feed the features of the test images to Random Forest Classifier to predict its class
result2 = SVM.predict(d2_image)
print(result2)

#text prediction
if result == 0:
    prediction = 'ball'
elif result ==1:
    prediction = 'bill'
elif result ==2:
    prediction = 'dog'
elif result ==3:
    prediction = 'flower'
elif result ==4:
    prediction = 'food'
    
   
print(prediction)

#image plotting
import cv2
img_path=r"C:\Users\hp\Desktop\neuroequilibrium-assignment-2\Assign-1-cnn-svm-rfc\data\validation\image_0051.jpg"
img = cv2.imread(img_path, cv2.IMREAD_COLOR)
img = cv2.resize(img, (227, 227))
img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
plt.imshow(img)
