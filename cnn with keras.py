
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
%matplotlib inline

import tensorflow as tf
#from tensorflow.keras.preprocessing.image import ImageDataGenerator
#from tensorflow.keras.preprocessing.image import img_to_array, load_img
from keras.models import Model, Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D,GlobalAveragePooling2D
from keras.layers import LSTM, Input, TimeDistributed,Convolution2D,Activation
from keras.layers.convolutional import ZeroPadding2D
from keras.optimizers import RMSprop, SGD
from keras.layers.normalization import BatchNormalization
from keras.utils import np_utils
from keras import optimizers
from keras.preprocessing import sequence
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.models import load_model

import random
import os
print(os.listdir(r"C:\Users\hp\Desktop\neuroequilibrium-assignment-2\Assign-1-cnn-svm-rfc\data"))


ball_dir = os.path.join(r"C:\Users\hp\Desktop\neuroequilibrium-assignment-2\Assign-1-cnn-svm-rfc\data\train\ClassBall")
bill_dir = os.path.join(r"C:\Users\hp\Desktop\neuroequilibrium-assignment-2\Assign-1-cnn-svm-rfc\data\train\ClassBill")
dog_dir = os.path.join(r"C:\Users\hp\Desktop\neuroequilibrium-assignment-2\Assign-1-cnn-svm-rfc\data\train\ClassDog")

flower_dir = os.path.join(r"C:\Users\hp\Desktop\neuroequilibrium-assignment-2\Assign-1-cnn-svm-rfc\data\train\ClassFlower")
food_dir = os.path.join(r"C:\Users\hp\Desktop\neuroequilibrium-assignment-2\Assign-1-cnn-svm-rfc\data\train\ClassFood")

print('Total training ball_dir images:', len(os.listdir(ball_dir)))
print('Total training bill_dir images:', len(os.listdir(bill_dir)))
print('Total training dog_dir images:', len(os.listdir(dog_dir)))

print('Total training flower_dir images:', len(os.listdir(flower_dir)))
print('Total training food_dir images:', len(os.listdir(food_dir)))

ball_files = os.listdir(ball_dir)
print(ball_files[:10])

bill_files = os.listdir(bill_dir)
print(bill_files[:10])

dog_files = os.listdir(dog_dir)
print(dog_files[:10])

flower_files = os.listdir(flower_dir)
print(flower_files[:10])

food_files = os.listdir(food_dir)
print(food_files[:10])



pic_index = 2

next_ball= [os.path.join(ball_dir, fname) 
                for fname in ball_files[pic_index-2:pic_index]]
next_bill = [os.path.join(bill_dir, fname) 
                for fname in bill_files[pic_index-2:pic_index]]
next_dog = [os.path.join(dog_dir, fname) 
                for fname in dog_files[pic_index-2:pic_index]]
next_flower = [os.path.join(flower_dir, fname) 
                for fname in flower_files[pic_index-2:pic_index]]

next_food = [os.path.join(food_dir, fname) 
                for fname in food_files[pic_index-2:pic_index]]




for i, img_path in enumerate(next_ball+next_bill+next_dog+next_flower+next_food):
    img = mpimg.imread(img_path)
    plt.imshow(img)
    plt.axis('Off')
    plt.show()
    
    
    
TRAINING_DIR = r"C:\Users\hp\Desktop\neuroequilibrium-assignment-2\Assign-1-cnn-svm-rfc\data\train"
training_datagen = ImageDataGenerator(
    rescale = 1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest')

VALIDATION_DIR = r"C:\Users\hp\Desktop\neuroequilibrium-assignment-2\Assign-1-cnn-svm-rfc\data\test"
validation_datagen = ImageDataGenerator(rescale = 1./255)

train_generator = training_datagen.flow_from_directory(
    TRAINING_DIR,
    target_size=(150,150),
    class_mode='categorical')

validation_generator = validation_datagen.flow_from_directory(
    VALIDATION_DIR,
    target_size=(150,150),
    class_mode='categorical')




#cnn layer
model = Sequential()
    # Note the input shape is the desired size of the image 150x150 with 3 bytes color
    # This is the first convolution
model.add(Conv2D(96, (3,3), activation='relu', input_shape=(150, 150, 3))),
model.add(MaxPooling2D(2, 2)),

# The second convolution
model.add(Conv2D(128, (3,3), activation='relu')),
model.add(MaxPooling2D(2,2)),
# The third convolution
model.add(Conv2D(192, (3,3), activation='relu')),
model.add(MaxPooling2D(2,2)),
# Flatten the results to feed into a DNN
model.add(Flatten()),
model.add(Dropout(0.5)),
# 512 neuron hidden layer
model.add(Dense(512, activation='relu')),
model.add(Dense(5, activation='softmax'))


    
#summary    
model.summary()


#model compile
model.compile(loss = 'categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
#model fitting
history = model.fit_generator(train_generator, epochs=25, validation_data = validation_generator, verbose = 1)

#saving modele
model.save("rps.h5")


#plotting accuracu validation and trainng
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'r', label='Training accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend(loc=0)
plt.figure()


plt.show()







#prediction on individual images
from keras.models import load_model
from keras.preprocessing import image
import numpy as np

# dimensions of our images    -----   are these then grayscale (black and white)?
img_width, img_height = 150, 150

# load the model we saved
model =  load_model("rps.h5")

path=r"C:\Users\hp\Desktop\neuroequilibrium-assignment-2\Assign-1-cnn-svm-rfc\data\validation\image_0055.jpg"

# Get test image ready
test_image = image.load_img(path, target_size=(img_width, img_height))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis=0)

test_image = test_image.reshape(1,img_width, img_height,3)    # Ambiguity!
# Should this instead be: test_image.reshape(img_width, img_height, 3) ??

result = model.predict_classes(test_image, batch_size=1)
print (result)


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
    
   
print("It's a class :",prediction)

#image plotting
import cv2
img = cv2.imread(path, cv2.IMREAD_COLOR)
img = cv2.resize(img, (227, 227))
img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
plt.imshow(img)


