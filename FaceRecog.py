#%%
import numpy as np
import pandas as pd
import cv2
import os

#%%
dataX=[]
dataY=[]
#%%
for x in os.listdir("C:\\Users\\albym\\VSCODE\\FaceRecogRandomForest\\FaceRecognitionRandomForest\\dataset\\dataset\\Andy"):
    temp=cv2.imread("C:\\Users\\albym\\VSCODE\\FaceRecogRandomForest\\FaceRecognitionRandomForest\\dataset\\dataset\\Andy\\"+x)
    temp=cv2.resize(temp,(64,64))
    dataX.append(temp)
    dataY.append('Andy')
#%%
for x in os.listdir("C:\\Users\\albym\\VSCODE\\FaceRecogRandomForest\\FaceRecognitionRandomForest\\dataset\\dataset\\Angela"):
    temp=cv2.imread("C:\\Users\\albym\\VSCODE\\FaceRecogRandomForest\\FaceRecognitionRandomForest\\dataset\\dataset\\Angela\\"+x)
    temp=cv2.resize(temp,(64,64))
    dataX.append(temp)
    dataY.append('Angela')
#%%
for x in os.listdir("C:\\Users\\albym\\VSCODE\\FaceRecogRandomForest\\FaceRecognitionRandomForest\\dataset\\dataset\\Creed"):
    temp=cv2.imread("C:\\Users\\albym\\VSCODE\\FaceRecogRandomForest\\FaceRecognitionRandomForest\\dataset\\dataset\\Creed\\"+x)
    temp=cv2.resize(temp,(64,64))
    dataX.append(temp)
    dataY.append('Creed')
#%%
for x in os.listdir("C:\\Users\\albym\\VSCODE\\FaceRecogRandomForest\\FaceRecognitionRandomForest\\dataset\\dataset\\Dwight"):
    temp=cv2.imread("C:\\Users\\albym\\VSCODE\\FaceRecogRandomForest\\FaceRecognitionRandomForest\\dataset\\dataset\\Dwight\\"+x)
    temp=cv2.resize(temp,(64,64))
    dataX.append(temp)
    dataY.append('Dwight')
#%%
for x in os.listdir("C:\\Users\\albym\\VSCODE\\FaceRecogRandomForest\\FaceRecognitionRandomForest\\dataset\\dataset\\Jim"):
    temp=cv2.imread("C:\\Users\\albym\\VSCODE\\FaceRecogRandomForest\\FaceRecognitionRandomForest\\dataset\\dataset\\Jim\\"+x)
    temp=cv2.resize(temp,(64,64))
    dataX.append(temp)
    dataY.append('Jim')
#%%
for x in os.listdir("C:\\Users\\albym\\VSCODE\\FaceRecogRandomForest\\FaceRecognitionRandomForest\\dataset\\dataset\\Kelly"):
    temp=cv2.imread("C:\\Users\\albym\\VSCODE\\FaceRecogRandomForest\\FaceRecognitionRandomForest\\dataset\\dataset\\Kelly\\"+x)
    temp=cv2.resize(temp,(64,64))
    dataX.append(temp)
    dataY.append('Kelly')
#%%
for x in os.listdir("C:\\Users\\albym\\VSCODE\\FaceRecogRandomForest\\FaceRecognitionRandomForest\\dataset\\dataset\\Kevin"):
    temp=cv2.imread("C:\\Users\\albym\\VSCODE\\FaceRecogRandomForest\\FaceRecognitionRandomForest\\dataset\\dataset\\Kevin\\"+x)
    temp=cv2.resize(temp,(64,64))
    dataX.append(temp)
    dataY.append('Kevin')
#%%
for x in os.listdir("C:\\Users\\albym\\VSCODE\\FaceRecogRandomForest\\FaceRecognitionRandomForest\\dataset\\dataset\\Meredith"):
    temp=cv2.imread("C:\\Users\\albym\\VSCODE\\FaceRecogRandomForest\\FaceRecognitionRandomForest\\dataset\\dataset\\Meredith\\"+x)
    temp=cv2.resize(temp,(64,64))
    dataX.append(temp)
    dataY.append('Meredith')
#%%
for x in os.listdir("C:\\Users\\albym\\VSCODE\\FaceRecogRandomForest\\FaceRecognitionRandomForest\\dataset\\dataset\\Michael"):
    temp=cv2.imread("C:\\Users\\albym\\VSCODE\\FaceRecogRandomForest\\FaceRecognitionRandomForest\\dataset\\dataset\\Michael\\"+x)
    temp=cv2.resize(temp,(64,64))
    dataX.append(temp)
    dataY.append('Michael')
#%%
for x in os.listdir("C:\\Users\\albym\\VSCODE\\FaceRecogRandomForest\\FaceRecognitionRandomForest\\dataset\\dataset\\Oscar"):
    temp=cv2.imread("C:\\Users\\albym\\VSCODE\\FaceRecogRandomForest\\FaceRecognitionRandomForest\\dataset\\dataset\\Oscar\\"+x)
    temp=cv2.resize(temp,(64,64))
    dataX.append(temp)
    dataY.append('Oscar')
#%%
for x in os.listdir("C:\\Users\\albym\\VSCODE\\FaceRecogRandomForest\\FaceRecognitionRandomForest\\dataset\\dataset\\Pam"):
    temp=cv2.imread("C:\\Users\\albym\\VSCODE\\FaceRecogRandomForest\\FaceRecognitionRandomForest\\dataset\\dataset\\Pam\\"+x)
    temp=cv2.resize(temp,(64,64))
    dataX.append(temp)
    dataY.append('Pam')
#%%
for x in os.listdir("C:\\Users\\albym\\VSCODE\\FaceRecogRandomForest\\FaceRecognitionRandomForest\\dataset\\dataset\\Phyllis"):
    temp=cv2.imread("C:\\Users\\albym\\VSCODE\\FaceRecogRandomForest\\FaceRecognitionRandomForest\\dataset\\dataset\\Phyllis\\"+x)
    temp=cv2.resize(temp,(64,64))
    dataX.append(temp)
    dataY.append('Phylis')
#%%
for x in os.listdir("C:\\Users\\albym\\VSCODE\\FaceRecogRandomForest\\FaceRecognitionRandomForest\\dataset\\dataset\\Rose"):
    temp=cv2.imread("C:\\Users\\albym\\VSCODE\\FaceRecogRandomForest\\FaceRecognitionRandomForest\\dataset\\dataset\\Rose\\"+x)
    temp=cv2.resize(temp,(64,64))
    dataX.append(temp)
    dataY.append('Rose')
#%%
denemeX=np.array(dataX)
denemeY=np.array(dataY)
denemeX=denemeX.reshape(3957,12288)

#%%
from sklearn.model_selection import train_test_split
xTrain,xTest,yTrain,yTest=train_test_split(denemeX,denemeY,test_size=0.15,random_state=1)
#%%
from sklearn.svm import SVC
svm=SVC(random_state=1)
svm.fit(xTrain,yTrain)
print("accuracy"+str(svm.score(xTest,yTest)))
#%%
from sklearn.ensemble import RandomForestClassifier
RF=RandomForestClassifier(n_estimators=1000,random_state=1)
RF.fit(xTrain,yTrain)
print("Random Forest Score: ",RF.score(xTest,yTest))
#%%
cv2.('im',dataX[10])
cv2.waitKey()
#just for testing
#%%
