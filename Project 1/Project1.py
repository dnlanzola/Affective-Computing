# ############################################################
# Name: Project1.py                                          #
# Authors: Blanette Baltimore, Karishma Jayaprakash,         #
# Curtis Smith III, Matthew Kramer and Daniel Anzola Delgado #
##############################################################

from __future__ import print_function
import os
import sys
import copy
import csv
import pdb
import numpy as np
from math import log, e
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC  
from sklearn.metrics import classification_report, confusion_matrix 

class Task:
	def __init__(self, ID, BP1, BP2, BP3, BP4, R1, R2, BPM, EDA, pain):
		
		'''Each subject performed tasks meant to elicit a specific emotion. For each task, their physiological data was recorded.  
		   The members of this class represent each measurement taken. The ID variable holds the number of the task. BP1 - BP4 
		   hold 4 diferent blood pressure rates measurements, R1 and R2 hold the subject's respiration rates, HR holds the 
		   subject's pulse rate, EDA holds the subject's electrodermal activity. The pain variable is there to signify whether or 
		   not the emotion elicited from the task was pain. 0 = No pain, 1 = Pain'''

		self.ID = ID		# Task Number

		# Each of the elements from BP1 to EDA is a list
		
		self.BP1 = BP1  	# BP Dia_mmHg
		self.BP2 = BP2		# BP_mmHg
		self.BP3 = BP3		# LA Mean BP_mmHg
		self.BP4 = BP4		# LA Systolic BP_mmHg
		self.R1 = R1			# Resp_Volts
		self.R2 = R2			# Respiration Rate_BPM
		self.BPM = BPM 		# Pulse Rate_BPM
		self.EDA = EDA		# EDA_microsiemens
		
		self.pain = pain	# Whether is Task 8 or not


class Subject:
	def __init__(self, ID, gender, arrTask, arrPain, arrNoPain):

		'''The datasets provided give only three pieces of information on each subject/participant - their ID number, their gender 
		   and the different tasks they performed. The Subject class reflects this in its members. The ID variable holds the ID number 
		   of the subject (e.g. F001's ID would be '001'), the gender variable holds the subject's gender and 'Task' is a list of Task
		   objects.'''

		self.ID = ID
		self.gender = gender
		self.arrTask = arrTask # Each subject participated in up to 10 tasks
		self.arrPain = arrPain
		self.arrNoPain = arrNoPain		

################################################ FUNCTION DEFINITIONS ######################################################
def entropy(labels, base=None):
  value,counts = np.unique(labels, return_counts=True)
  norm_counts = counts / counts.sum()
  base = e if base is None else base
  return -(norm_counts * np.log(norm_counts)/np.log(base)).sum()

def calculations(arr):
	'''The calculations function will perform all calculations needed in order to get the 5 features from
	   each physiological measurement.'''

	mean = float(sum(arr)) / max(len(arr),1)
	variance = sum((xi - mean) ** 2 for xi in arr) / len(arr)
	ent = entropy(arr)
	minVal = min(arr)
	maxVal = max(arr)
	
	features = [mean, variance, ent, minVal, maxVal]

	return features

def random_forest_classifier(features, target):
	clf = RandomForestClassifier()
	clf.fit(features, target)
	return clf

path = "/data/scanavan1/AffectiveComputing/Project1"

# Reading arguments from command line input
algorithm = sys.argv[1]
dataset = sys.argv[2]

headers = ["min", "max", "mean", "entropy", "variance"]

# Appending desired dataset to end of path
if dataset == "1":
    path += "/Dataset1"
else:
    path += "/Dataset2"

################################################## READING DATA ######################################################

folders = [] # List to hold names of all subject folders in current directory
subjects = [] # List of subject objects

 # Reading names of the folders in current directory
files = os.listdir(path)

for name in files:
	folders.append(name)

folders.sort() # Alphabetically sorting folders

taskFolders = [] # List to hold task folders
i = 0

auxArr = []
auxT = Task(0,1,2,3,4,5,6,7,8,9)
auxS = Subject(0,"N",auxArr,auxArr,auxArr)

arrSubject = [] # 2D List for feature vectors
arrLabel = [] # 1D List for labels of feature vectors (P/NP)

arrP = []
arrNP = []

# FILE NAVIGATION
i = 0
while i < len(folders):
	# Creates an object with random placeholder values
	auxS = Subject(0,"N",auxArr,auxArr,auxArr)
	
	# Subject ID assignment
	auxS.ID = str(folders[i])

	# Gender assignment
	if "F" in folders[i]:
		auxS.gender = "F"
	else:
		auxS.gender = "M"

	auxTaskArray = []


	subpath = path + "/" + folders[i] 	# Path to access the task folders
	files = os.listdir(subpath)
	
	# "taskFolders": List of task folders (T1, T2, T3,...)
	taskFolders = []

	for name in files:
		taskFolders.append(name)

	# "dataFilesList": list of text files containing the physiological data  "BP Dia_mmHg.txt", "BP_mmHg.txt", etc.
	dataFilesList = []


	
	j = 0
	while (j < len(taskFolders)):

		# Assigns the task number to the ID member of the task object (e.g. "T1")
		auxT.ID = taskFolders[j]

		taskpath = subpath + "/" + taskFolders[j] # Path to access the list of data .txt files
		files = os.listdir(taskpath)
		
		for name in files:
			dataFilesList.append(name)
		
		auxFeatureVector = [] # Holds all 40 features for the 8 .txt files in each task folder
		auxT = Task(0,1,2,3,4,5,6,7,8,9)

		k = 0
		while (k < len(dataFilesList)):
			
			# REMOVE NEW LINE CHARACTER
			lines = [line.rstrip('\r\n') for line in open(taskpath + "/" + dataFilesList[k])] # Removes newline from end of string
			
			# CONVERT STRING TO FLOAT
			floatLines = []
			for item in lines:
				floatLines.append(float(item))

			auxFV = [] # Gets five features for each .txt file
			
			if (dataFilesList[k] == "BP Dia_mmHg.txt"):	
				auxFV = calculations(floatLines)
				l = 0
				while (l < len(auxFV)):
					auxFeatureVector.append(auxFV[l])
					l = l + 1
				auxT.BP1 = floatLines
			
			if (dataFilesList[k] == "BP_mmHg.txt"):	
				auxFV = calculations(floatLines)
				l = 0
				while (l < len(auxFV)):
					auxFeatureVector.append(auxFV[l])
					l = l + 1
				auxT.BP2 = floatLines
			
			if (dataFilesList[k] == "LA Mean BP_mmHg.txt"):	
				auxFV = calculations(floatLines)
				l = 0
				while (l < len(auxFV)):
					auxFeatureVector.append(auxFV[l])
					l = l + 1
				auxT.BP3 = floatLines

			if (dataFilesList[k] == "LA Systolic BP_mmHg.txt"):	
				auxFV = calculations(floatLines)
				l = 0
				while (l < len(auxFV)):
					auxFeatureVector.append(auxFV[l])
					l = l + 1
				auxT.BP4 = floatLines

			if (dataFilesList[k] == "Resp_Volts.txt"):	
				auxFV = calculations(floatLines)
				l = 0
				while (l < len(auxFV)):
					auxFeatureVector.append(auxFV[l])
					l = l + 1
				auxT.R1 = floatLines

			if (dataFilesList[k] == "Respiration Rate_BPM.txt"):
				auxFV = calculations(floatLines)
				l = 0
				while (l < len(auxFV)):
					auxFeatureVector.append(auxFV[l])
					l = l + 1
				auxT.R2 = floatLines															

			if (dataFilesList[k] == "Pulse Rate_BPM.txt"):
				auxFV = calculations(floatLines)
				l = 0
				while (l < len(auxFV)):
					auxFeatureVector.append(auxFV[l])
					l = l + 1
				auxT.BPM = floatLines

			if (dataFilesList[k] == "EDA_microsiemens.txt"):
				auxFV = calculations(floatLines)
				l = 0
				while (l < len(auxFV)):
					auxFeatureVector.append(auxFV[l])
					l = l + 1
				auxT.EDA = floatLines
			
		
			k = k + 1
		
		
		auxTaskArray.append(auxT)




		'''NB: We need a 2D list and a 1D list to pass into the classifier. The 2D list will be of the features
		   and the 1D list is a list of labels. The labels will be either "P" or "NP" which stand for "pain" and
		   "no pain" respectively. Each feature vector in the 2D list will have a corresponding label in the 1D
		   list. Therefore a feature vector's label will be located at the same index in a 1D list as the feature
		   vector in the 2D list. Columns = Features, Rows = Subjects'''

		if "8" in taskFolders[j]:
			auxT.pain = 1
			auxS.arrPain = list(auxFeatureVector)
			arrSubject.append(auxFeatureVector) # Appending a list to a list creates a 2D list
			arrP.append(auxFeatureVector)
			arrLabel.append("P")
		else:
			auxT.pain = 0
			auxS.arrNoPain = list(auxFeatureVector)
			arrSubject.append(auxFeatureVector)
			arrNP.append(auxFeatureVector)
			arrLabel.append("NP")

		auxS.arrTask = list(auxTaskArray)

		dataFilesList = []
		j = j + 1
		
		
	subjects.append(auxS)
	i = i + 1				




####################################### SPLITTING DATA ######################################################

# GOAL: Split list of subjects into two lists: 80% of subjects = training, 20% of subjects = testing

train_x = []
train_y = []
test_x = []
test_y = []
boundTrain = 0.8 * len(arrSubject)
boundTest = 0.2 * len(arrSubject)

i = 0

# The while loop below takes 80% of the subjects and adds them to the training list

while (i < len(arrSubject)):
	if (i < boundTrain):
		train_x.append(arrSubject[i])
		train_y.append(arrLabel[i])
	else:
		test_x.append(arrSubject[i])
		test_y.append(arrLabel[i])
	i += 1


def avgArr(arrP):
	arrMeanP = []
	arrPreMeanP = []

	i = 0
	while (i < len(arrP)):
		j = 0
		auxArr = []
		a = arrP[i][0]
		b = arrP[i][1]
		c = arrP[i][2]
		d = arrP[i][3]
		e = arrP[i][4]
		while (j < 4):
			a = a + arrP[i][j+5]
			b = b + arrP[i][(j+1)+5]
			c = c + arrP[i][(j+2)+5]
			d = d + arrP[i][(j+3)+5]
			e = e + arrP[i][(j+4)+5]
			j = j + 1
		auxArr.append(a/(len(arrP[i])/5))
		auxArr.append(b/(len(arrP[i])/5))
		auxArr.append(c/(len(arrP[i])/5))
		auxArr.append(d/(len(arrP[i])/5))
		auxArr.append(e/(len(arrP[i])/5))
		arrPreMeanP.append(auxArr)

		i = i + 1
		
	# print ("ARR PRE MEAN CORRELATION")
	# print (arrPreMeanP)

	i = 0
	a = 0
	b = 0
	c = 0
	d = 0
	e = 0
	while ( i < len(arrPreMeanP)):
		a = a + arrPreMeanP[i][0]
		b = b + arrPreMeanP[i][1]
		c = c + arrPreMeanP[i][2]
		d = d + arrPreMeanP[i][3]
		e = e + arrPreMeanP[i][4]

		i = i + 1

	a = a / len(arrPreMeanP)
	b = b / len(arrPreMeanP)
	c = c / len(arrPreMeanP)
	d = d / len(arrPreMeanP)
	e = e / len(arrPreMeanP)

	arrMeanP.append(a)
	arrMeanP.append(b)
	arrMeanP.append(c)
	arrMeanP.append(d)
	arrMeanP.append(e)

	return arrMeanP

####################################### XAI ######################################################
# 	features = [mean, variance, ent, minVal, maxVal]


arrP = []
arrNP = []

i = 0
while ( i < len(train_x)):
	if (train_y[i] == "P"):
		arrP.append(train_x[i])
	else:
		arrNP.append(train_x[i])

	i = i + 1

trainXPMean = avgArr(arrP)
trainXNPMean = avgArr(arrNP)

print ("Train_x Pain Mean")
print (trainXPMean)
print ("Train_x No Pain Mean")
print (trainXNPMean)


arrP = []
labelsP = []
arrNP = []
labelsNP = []

i = 0
while ( i < len(test_x)):
	if (test_y[i] == "P"):
		arrP.append(test_x[i])
		labelsP.append(1.000)
	else:
		arrNP.append(test_x[i])
		labelsNP.append(0.000)

	i = i + 1

testXPMean = avgArr(arrP)
testXNPMean = avgArr(arrNP)

print ("Test_x Pain Mean")
print (testXPMean)
print ("Test_x No Pain Mean")
print (testXNPMean)

i = 0

# The code below will use numpy to calculate the correlation between two vectors
# NB: Ask Professor Canavan if this is the correlation we were supposed to be calculating

print ("[PAIN] Correlation Train x & Test X")
print (np.corrcoef(trainXPMean, testXPMean))

print ("[NO PAIN] Correlation Train x & Test X")
print(np.corrcoef(trainXNPMean, testXNPMean))

explanation = 0



###################################################### CLASSIFYING DATA #######################################################
############################################################ & ################################################################
########################################################### XAI ###############################################################
if algorithm == "RF":
	print ("--- Random Forest ---")
	trained_model = [[]]
	# Create random forest classifier instance
	trained_model = random_forest_classifier(train_x, train_y)

	predictions = trained_model.predict(test_x)

	for i in range(0, 5):
		print ("Actual outcome :: {} and Predicted outcome :: {}".format(list(test_y)[i], predictions[i]) )
		
	print ("Train Accuracy :: ", accuracy_score(train_y, trained_model.predict(train_x)))
	print ("Test Accuracy  :: ", accuracy_score(test_y, predictions))
	print (" Confusion matrix ", confusion_matrix(test_y, predictions))

	explanation = (accuracy_score(test_y, predictions)) / 100
	inc = 100 - explanation

	print("Explanation Accuracy: ", inc)
	print("Incorrect Explanation Accuracy: ", explanation)
	
else:
	print ("--- SVM ---")
	svclassifier = SVC(kernel='linear')  
	svclassifier.fit(train_x, train_y) 
	y_pred = svclassifier.predict(test_x) 
	print(confusion_matrix(test_y,y_pred))  
	print(classification_report(test_y,y_pred))
