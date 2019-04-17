# ############################################################
# Name: Project3.py                                          #
# Authors: Blanette Baltimore, Karishma Jayaprakash,         #
# Curtis Smith III, Matthew Kramer and Daniel Anzola Delgado #
##############################################################

from __future__ import print_function
import os
import sys
import copy
import cv2
import numpy as np 
from math import log, e
import pandas as pd
import sklearn.metrics

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
from sklearn.metrics import classification_report
from keras.layers.normalization import BatchNormalization
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import np_utils

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC  
from sklearn.metrics import classification_report, confusion_matrix 



class Task:
	def __init__(self, ID, imageName, image, landmarkName, landmark, pain):
		
		'''Each subject performed tasks meant to elicit a specific emotion. For each task, their physiological data was recorded.  
		   The members of this class represent each measurement taken. The ID variable holds the number of the task. BP1 - BP4 
		   hold 4 diferent blood pressure rates measurements, R1 and R2 hold the subject's respiration rates, HR holds the 
		   subject's pulse rate, EDA holds the subject's electrodermal activity. The pain variable is there to signify whether or 
		   not the emotion elicited from the task was pain. 0 = No pain, 1 = Pain'''

		self.ID = ID # Task Number
		
		# Arrays containing data
		self.imageName = imageName
		self.image = image 
		self.landmarkName = landmarkName
		self.landmark = landmark	

		self.pain = pain # Whether is Task 8 or not


class Subject:
	def __init__(self, ID, gender, arrTask, FV):

		'''The datasets provided give only three pieces of information on each subject/participant - their ID number, their gender 
		   and the different tasks they performed. The Subject class reflects this in its members. The ID variable holds the ID number 
		   of the subject (e.g. F001's ID would be '001'), the gender variable holds the subject's gender and 'Task' is a list of Task
		   objects.'''

		self.ID = ID
		self.gender = gender
		self.arrTask = arrTask # Each subject participated in up to 10 tasks

		self.FV = FV # Feature vector

		# Arrays containing data, classified pain
		# self.imagePain = imagePain
		# self.imageNoPain = imageNoPain
		# self.landmarksPain = landmarksPain
		# self.landmarksNoPain = landmarksNoPain		
			
################################################## READING DATA ######################################################

numberPain = 0
numberNoPain = 0


path  = "/data/scanavan1/BP4D+/2D+3D"

folders = [] # List to hold names of all subject folders in current directory
subjects = [] # List of subject objects

# Reading names of the folders in current directory
files = os.listdir(path)

for name in files:
	folders.append(name) # List of all the folders in the main folders

folders.sort() # Alphabetically sorting folders

taskFolders = [] # List to hold task folders

i = 0

auxArr = []
auxT = Task(0,auxArr,auxArr,auxArr,auxArr,3)
auxS = Subject(0,"N", auxArr, auxArr)


arrSubject = [] # 2D List for feature vectors
arrLabel = [] # 1D List for labels of feature vectors (P/NP)

################# PROCESSING LANDMARKS #################

pathLandmarks = "/data/scanavan1/BP4D+/3DFeatures/FacialLandmarks"

landmarks = [] # Holds list of landmarks

filesLandmarks = os.listdir(pathLandmarks) # Reading list of landmarks in current directory

for name in filesLandmarks:
	landmarks.append(name)


landmarks.sort()

for name in landmarks:
	print(name)

arrLandmarks = []

landSubjects = []
landTasks = []
landImg = []

i = 0
while (i < len(landmarks)):
	data = landmarks[i].split("_")
	data[2] = data[2][:-8]		# Removes ".bndplus" extension from third element (image name)
	arrLandmarks.append(data)

	if (data[0] not in landSubjects):
		landSubjects.append(data[0])

	i = i + 1	

################# FILE NAVIGATION #################

pathOriginal  = "/data/scanavan1/BP4D+/2D+3D"

xSubjects = []

landTasks = []
landImg = []
print("LANDSUBJECTS")
i = 0
while (i < len(landSubjects)):
	# Empty object type Subject
	auxS = Subject(0,"N", auxArr, auxArr)

	# Subject ID assignation
	auxS.ID = landSubjects[i]

	j = 0
	landTasks = []
	auxTasks = []
	while(j < len(arrLandmarks)):
		
		# Empty object type Task
		auxT = Task(0,auxArr,auxArr,auxArr,auxArr,3)

		if (arrLandmarks[j][0] == landSubjects[i]):
			if(arrLandmarks[j][1] not in landTasks):
				landTasks.append(arrLandmarks[j][1])
				
				# Task ID assignation
				auxT.ID = arrLandmarks[j][1]
				
				# print(arrLandmarks[j][1])

				landImg = []
				auxLandmark = []
				
				counter = 0
				k = 0
				while (k < len(arrLandmarks)):
					bndData = []
					if ((arrLandmarks[k][0] == landSubjects[i]) and (arrLandmarks[j][1] == arrLandmarks[k][1])):


						if (arrLandmarks[j][1] == "T8"):
							if (counter < 72):  #72
								#print(arrLandmarks[k][2])
								
								landImg.append(arrLandmarks[k][2])

								# print("\n***")
								# print("Subject: ", landSubjects[i])
								# print("Task: ", arrLandmarks[j][1])
								# print("Landmark: ", arrLandmarks[k][2])
								# print("***\n")

								# CONSTRUCTING BND FILE PATH: "/data/scanavan1/BP4D+/3DFeatures/FacialLandmarks/F001_T1_0001.bndplus"
								bndPath = pathLandmarks + "/" + landSubjects[i] + "_" + arrLandmarks[j][1] + "_" + arrLandmarks[k][2] + ".bndplus"
								
								#print ("bndPath: ", bndPath)
								
								# HERE WE READ & APPEND THE LANDMARKS TO ATTRIBUTE landmark
								bndFile = open(bndPath)
								
								# Each line of bndplus files contains 3 coordinates (x, y, z)
								# The newline char is deleted, it is split by comma, and the 3 numbers are appended to bndData
								for line in bndFile.readlines():
									line = line.strip()
									line = line.split(",")
									bndData.append(line[0])
									bndData.append(line[1])
									bndData.append(line[2])

								# Close file
								bndFile.close()
								
								auxLandmark.append(bndData)

								counter = counter + 1
								numberPain = numberPain + 1
							else:
								break
						else:
							if (counter < 24):  #24
								#print(arrLandmarks[k][2])
								
								landImg.append(arrLandmarks[k][2])
								
								# print("\n***")
								# print("Subject: ", landSubjects[i])
								# print("Task: ", arrLandmarks[j][1])
								# print("Landmark: ", arrLandmarks[k][2])
								# print("***\n")
								# CONSTRUCTING BND FILE PATH: "/data/scanavan1/BP4D+/3DFeatures/FacialLandmarks/F001_T1_0001.bndplus"

								bndPath = pathLandmarks + "/" + landSubjects[i] + "_" + arrLandmarks[j][1] + "_" + arrLandmarks[k][2] + ".bndplus"
								
								#print ("bndPath: ", bndPath)
								
								# HERE WE READ & APPEND THE LANDMARKS TO ATTRIBUTE landmark
								bndFile = open(bndPath, "r")
								
								# Each line of bndplus files contains 3 coordinates (x, y, z)
								# The newline char is deleted, it is split by comma, and the 3 numbers are appended to bndData								
								for line in bndFile.readlines():
									line = line.strip()
									line = line.split(",")
									bndData.append(line[0])
									bndData.append(line[1])
									bndData.append(line[2])

								# Close file
								bndFile.close()
								
								auxLandmark.append(bndData)

								counter = counter + 1	
								numberNoPain = numberNoPain + 1				
							else:
								break
					
					
					k = k + 1

				auxT.landmarkName = list (landImg)
				auxT.landmark = list (auxLandmark)
				auxTasks.append(auxT)
		
		
		j = j + 1


	auxS.arrTask = list (auxTasks)
	xSubjects.append(auxS)

	i = i + 1





################# ADDING IMAGES TO OBJECTS #################

pathOriginal  = "/data/scanavan1/BP4D+/2D+3D"
failed = 0

i = 0
while i < len(xSubjects):
	
	# This will result in something like "/data/scanavan1/BP4D+/2D+3D/F001"
	path1 = pathOriginal + "/" + xSubjects[i].ID

	j = 0
	while j < len(xSubjects[i].arrTask):

		# This will result in something like "/data/scanavan1/BP4D+/2D+3D/F001/T1"
		path2 = path1 + "/" + xSubjects[i].arrTask[j].ID

		k = 0
		auxImageName = []
		auxImage = []
		while k < len(xSubjects[i].arrTask[j].landmarkName):

			# This will result in something like "/data/scanavan1/BP4D+/2D+3D/F001/T1/0001.jpg"
			# Some landmark files have an extra '0' on the image name. This fixes that issue
			exists = os.path.isfile(path2 + "/" + xSubjects[i].arrTask[j].landmarkName[k] + ".jpg")
			if exists:
				path3 = path2 + "/" + xSubjects[i].arrTask[j].landmarkName[k] + ".jpg"
			else:
				fixName = xSubjects[i].arrTask[j].landmarkName[k][1:]
				path3 = path2 + "/" + fixName + ".jpg"

			# Appends image name to imageName array
			auxImageName.append(xSubjects[i].arrTask[j].landmarkName[k])
			
			# HERE THE IMAGE WILL BE READ 
			img = cv2.imread(path3)
			img = cv2.resize(img,(256,256))
			img = img_to_array(img)

			auxImage.append(img)

			k = k + 1

		xSubjects[i].arrTask[j].imageName = list (auxImageName)
		xSubjects[i].arrTask[j].image = list(auxImage)
		j = j + 1

	i = i + 1


################# PRINT INFO #################
## This will be deleted for final submission##
print("NUMBER PAIN: ", numberPain)
print("NUMBER NO PAIN: ", numberNoPain)

print("\n")

i = 0
while ( i < len(xSubjects)):
	print("ID: ", xSubjects[i].ID)

	j = 0
	while (j < len(xSubjects[i].arrTask)):
		print("Task ID: ", xSubjects[i].arrTask[j].ID)
		print("Landmark Names: ", xSubjects[i].arrTask[j].landmarkName)
		print("Landmarks: ", xSubjects[i].arrTask[j].landmark)

		print("Image Names: ", xSubjects[i].arrTask[j].imageName)

		j = j + 1
	print("\n")
	i = i + 1

		
###############################################


