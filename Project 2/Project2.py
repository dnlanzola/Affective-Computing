##############################################################
# Name: Project2.py                                          #
# Authors: Blanette Baltimore, Karishma Jayaprakash,         #
# Curtis Smith III, Matthew Kramer and Daniel Anzola Delgado #
##############################################################

from __future__ import print_function
import os
import sys
import copy
import cv2
import numpy as np 

import keras
from keras import optimizers
from keras.optimizers import Adam
from keras.models import Sequential
from keras.layers.core import Dense
from keras.layers.core import Flatten
from keras.layers.core import Dropout
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras import backend as K
from sklearn.preprocessing import LabelBinarizer
from keras.layers.normalization import BatchNormalization
from keras.preprocessing.image import img_to_array
from keras.utils import np_utils

BATCH_SIZE = 64
EPOCHS = 10
CLASSES = 2 # Pain and No Pain
LR = 0.0001 # Learning rate

# Path to access the datasets and retrieve the image paths
path = "/data/scanavan1/AffectiveComputing/Project2/pain_classification"


# Reading names of the folders in current directory
files = os.listdir(path)


folders = [] # Will be one of the three main folders (Testing, Training, Validation)
folders2 = [] # Will be one of the subfolders -- Pain or No Pain
folders3 = [] # List of images in either the No Pain or the Pain folder


for name in files:
	folders.append(name)

folders.sort() # Alphabetically sorting folders


# DATA ARRAYS
testing = []
training = []
validation = []

# LABEL ARRAYS
testingLabel = []
trainingLabel = []
validationLabel = []


###################################### DATA READING AND AUGMENTATION ###########################################

'''The nested while loop below is used to  access the images from the three datasets (training, testing and 
   validation). The 'i' variable is used to iterate through the first level of folders, which are the Testing,
   Training and Validation folders. Once one of these folders is accessed, the 'j' variable is used to iterate
   through the final level of folders (Pain, No Pain). Once the final level of folders is reached, the contents
   of either the Pain or No Pain folder will contain an amount of images. The 'k' variable is used to index each
   image.'''

i = 0 # Testing, Training or Validation
while (i < len(folders)):
    path2 = path + "/" + folders[i]
    files2 = os.listdir(path2)

    folders2 = []
    for name in files2:
        folders2.append(name)

    folders2.sort()
    

    j = 0 # Pain or No pain
    while (j < len(folders2)):
        path3 = path2 + "/" + folders2[j]
        files3 = os.listdir(path3)

        folders3 = []
        for name in files3:
            folders3.append(name)

        
        folders3.sort()


        k = 0 # Used to index each image in the folder. Value willl increment for the amount of images available
        while (k < len(folders3)):

            if (i == 0 and j == 0):
                image = cv2.imread(path+"/"+folders[i]+"/"+folders2[j]+"/"+folders3[k])
                
                rows, cols = image.shape[:2]
                rotationMatrix = cv2.getRotationMatrix2D((cols/2, rows/2), 180, 1)
                rotatedIm =  cv2.warpAffine(image, rotationMatrix, (cols, rows))

                image = img_to_array(image)
                testing.append(image)
                testingLabel.append(0)

                rotatedIm = img_to_array(rotatedIm)
                testing.append(rotatedIm)
                testingLabel.append(0)


            if (i == 0 and j == 1):
                image = cv2.imread(path+"/"+folders[i]+"/"+folders2[j]+"/"+folders3[k])
                
                rows, cols = image.shape[:2]
                rotationMatrix = cv2.getRotationMatrix2D((cols/2, rows/2), 180, 1)
                rotatedIm =  cv2.warpAffine(image, rotationMatrix, (cols, rows))
                
                
                image = img_to_array(image)
                testing.append(image)
                testingLabel.append(1)   

                          
                rotatedIm = img_to_array(rotatedIm)
                testing.append(rotatedIm) 
                testingLabel.append(1)                               
          

            if (i == 1 and j == 0):
                image = cv2.imread(path+"/"+folders[i]+"/"+folders2[j]+"/"+folders3[k])
                
                rows, cols = image.shape[:2]
                rotationMatrix = cv2.getRotationMatrix2D((cols/2, rows/2), 180, 1)
                rotatedIm =  cv2.warpAffine(image, rotationMatrix, (cols, rows))
                
                
                image = img_to_array(image)
                training.append(image)
                trainingLabel.append(0)

                
                rotatedIm = img_to_array(rotatedIm)
                training.append(rotatedIm)   
                trainingLabel.append(0)                             
                              

            if (i == 1 and j == 1):
                image = cv2.imread(path+"/"+folders[i]+"/"+folders2[j]+"/"+folders3[k])
                
                rows, cols = image.shape[:2]
                rotationMatrix = cv2.getRotationMatrix2D((cols/2, rows/2), 180, 1)
                rotatedIm =  cv2.warpAffine(image, rotationMatrix, (cols, rows))

                
                image = img_to_array(image)
                training.append(image)
                trainingLabel.append(1)    

                           
                rotatedIm = img_to_array(rotatedIm)
                training.append(rotatedIm)
                trainingLabel.append(1)                                 
                              

            if (i == 2 and j == 0):
                image = cv2.imread(path+"/"+folders[i]+"/"+folders2[j]+"/"+folders3[k])
                
                rows, cols = image.shape[:2]
                rotationMatrix = cv2.getRotationMatrix2D((cols/2, rows/2), 180, 1)
                rotatedIm =  cv2.warpAffine(image, rotationMatrix, (cols, rows))
                
                
                image = img_to_array(image)
                validation.append(image)
                validationLabel.append(0)

                
                rotatedIm = img_to_array(rotatedIm)
                validation.append(rotatedIm)
                validationLabel.append(0)                               
                              

            if (i == 2 and j == 1):
                image = cv2.imread(path+"/"+folders[i]+"/"+folders2[j]+"/"+folders3[k])
                
                rows, cols = image.shape[:2]
                rotationMatrix = cv2.getRotationMatrix2D((cols/2, rows/2), 180, 1)
                rotatedIm =  cv2.warpAffine(image, rotationMatrix, (cols, rows))
                
                
                image = img_to_array(image)
                validation.append(image)
                validationLabel.append(1)                
                
                
                rotatedIm = img_to_array(rotatedIm)
                validation.append(rotatedIm)
                validationLabel.append(1)                                  
           
                                 
            k = k + 1

        j = j + 1

    i = i + 1

###################################### DATA PREPROCESSING ###########################################

'''The code below converts the data arrays to numpy arrays (np.array(dataset)),
   scales pixel intensities between 0 and 1 (dataset/= 255) and performs
   one-hot-encoding for the data label arrays (np_utils.to_categorical(dataset)).'''

# Creating numpy arrays
training = np.array(training).astype('float32')
trainingLabel = np.array(trainingLabel)

testing = np.array(testing).astype('float32')
testingLabel = np.array(testingLabel)

validation = np.array(validation).astype('float32')
validationLabel = np.array(validationLabel)

# Scaling pixel intensities between 0 and 1
training /= 255.0
testing /= 255.0
validation /= 255.0

# One-hot-encoding of labels
testingLabel = np_utils.to_categorical(testingLabel)
trainingLabel = np_utils.to_categorical(trainingLabel)
validationLabel = np_utils.to_categorical(validationLabel)
######################################## NEURAL NETWORK ###########################################


# Optimizer: SGD
# Activation: relu

if K.image_data_format() == 'channels_first':
    input_shape = (3, rows, cols)
else:
    input_shape = (rows, cols, 3)


model = Sequential()

model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))

# Last layer
model.add(Dropout(0.5))
model.add(Dense(CLASSES, activation='softmax'))

OPTIMIZER = 'SGD'

opt = optimizers.SGD(lr=LR, decay=0, momentum=0, nesterov=False)
model.compile(loss=keras.losses.categorical_crossentropy, optimizer=opt, metrics=["accuracy"])

model.fit(training, trainingLabel, batch_size=BATCH_SIZE, epochs=EPOCHS, verbose=1, validation_data=(validation, validationLabel))
score = model.evaluate(testing, testingLabel, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
print('SIZE OF SCORE: ', score.size)

print("BATCH SIZE: ", BATCH_SIZE)
print("EPOCHS: ", EPOCHS)
print("LEARNING RATE: ", LR)
print("OPTIMIZER: ", OPTIMIZER)
