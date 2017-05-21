# -*- coding: utf-8 -*-
"""
Created on Fri Apr 28 20:14:17 2017

@author: Pritam
"""

import pandas as pd
import numpy as np
import math
import time
from scipy.sparse import *
from sklearn import datasets
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier
from pandas import DataFrame
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.svm import SVC

import warnings
warnings.filterwarnings("ignore")

#Reading the train and test data from the file and storing it in an object
trainDataUberAug = pd.read_csv('uber-raw-data-aug14.csv', sep=',',header=0,keep_default_na=False)
testDataUberSep = pd.read_csv('uber-raw-data-sep14.csv', sep=',',header=0,keep_default_na=False)
trainDataLyftAug = pd.read_csv('lyft-raw-data-aug14.csv', sep=',',header=0,keep_default_na=False)
testDataLyftSep = pd.read_csv('lyft-raw-data-sep14.csv', sep=',',header=0,keep_default_na=False)
print(trainDataUberAug.shape)
print(trainDataUberAug.shape)
trainDataUberAug = trainDataUberAug.dropna()
trainDataLyftAug = trainDataLyftAug.dropna()
testDataLyftSep = testDataLyftSep.dropna()
testDataUberSep = testDataUberSep.dropna()
#trainDataLyft = trainDataLyft['Lat'].replace(to_replace="[a-zA-Z]", value='40.77398', regex=True, inplace=True)
#trainDataLyft = trainDataLyft['Lon'].replace(to_replace="[a-zA-Z]", value='-73.97848', regex=True, inplace=True)
#print(trainDataAug.shape)
#print(trainDataSep.shape)
#print(trainDataLyft.shape)
print('data read!!')

#frames = [trainDataAug, trainDataSep, trainDataLyft]

#trainData = pd.concat(frames)
#trainData = trainDataLyft
trainData = trainDataLyftAug.append(trainDataUberAug)
testData = testDataLyftSep.append(testDataUberSep)
#testData = testDataUberSep
#trainData.append(trainDataSep)
#trainData.append(trainDataLyft)
trainData["taxi_code"] = trainData['base_code'].map({'B02512':1,'B02598':2,'B02617':3,'B02682':4,'B02764':5,'B02765':6,'Lyft':7})
trainData = trainData.dropna()
testData["taxi_code"] = testData['base_code'].map({'B02512':1,'B02598':2,'B02617':3,'B02682':4,'B02764':5,'B02765':6,'Lyft':7})
testData = testData.dropna()


trainData['Date/Time'] = pd.to_datetime(trainData['Date/Time'], infer_datetime_format =True)
#print(trainData['Date/Time'][1])
trainData['hour'] = trainData['Date/Time'].dt.hour
#print(trainData['hour'][1])
trainData['Lat'] = trainData['Lat'].astype(int)
trainData['Lon'] = trainData['Lon'].astype(int)
testData['Date/Time'] = pd.to_datetime(testData['Date/Time'], infer_datetime_format =True)
#print(testData['Date/Time'][1])
testData['hour'] = testData['Date/Time'].dt.hour
#print(testData['hour'][1])
#trainData = trainData[trainData.columns[0:7]]

#trainData['Lat'].replace(to_replace="[a-zA-Z]", value='40.7748', regex=True, inplace=True)
#trainData['Lon'].replace(to_replace="[a-zA-Z]", value='-73.9914', regex=True, inplace=True)

#testData = pd.read_csv('augsepuber.csv', sep=',',header=0)
#testData.shape

#testData["taxi_code"] = testData['Taxi Service'].map({'Uber':1,'Lyft':2})
#testData = testData.dropna()
trainData.isnull().sum().sum()
#print(trainData.shape)
print(trainData.shape)
trainData=trainData.reset_index()

print("Model is -")
print("Prediction is - ")

#from sklearn.model_selection import train_test_split

X = trainData[["Lat", "Lon", "hour"]]
y = trainData[["taxi_code"]]
X_test = testData[["Lat", "Lon", "hour"]]
Y_test = testData[["taxi_code"]]
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.6, random_state=42)


#Plot Graph
fig, ax = plt.subplots()
index = np.arange(0,9600,400)
bar_width = 50.0
opacity = 0.8

Base1 = (len(trainData[trainData['hour']==0][trainData['taxi_code']== 1]), len(trainData[trainData['hour']==1][trainData['taxi_code']== 1]), len(trainData[trainData['hour']==2][trainData['taxi_code']== 1]), len(trainData[trainData['hour']==3][trainData['taxi_code']== 1]),len(trainData[trainData['hour']==4][trainData['taxi_code']== 1]),len(trainData[trainData['hour']==5][trainData['taxi_code']== 1]),len(trainData[trainData['hour']==6][trainData['taxi_code']== 1]),len(trainData[trainData['hour']==7][trainData['taxi_code']== 1]),len(trainData[trainData['hour']==8][trainData['taxi_code']== 1]),len(trainData[trainData['hour']==9][trainData['taxi_code']== 1]),len(trainData[trainData['hour']==10][trainData['taxi_code']== 1]),len(trainData[trainData['hour']==11][trainData['taxi_code']== 1]),len(trainData[trainData['hour']==12][trainData['taxi_code']== 1]),len(trainData[trainData['hour']==13][trainData['taxi_code']== 1]), len(trainData[trainData['hour']==14][trainData['taxi_code']== 1]),len(trainData[trainData['hour']==15][trainData['taxi_code']== 1]),len(trainData[trainData['hour']==16][trainData['taxi_code']== 1]),len(trainData[trainData['hour']==17][trainData['taxi_code']== 1]),len(trainData[trainData['hour']==18][trainData['taxi_code']== 1]),len(trainData[trainData['hour']==19][trainData['taxi_code']== 1]),len(trainData[trainData['hour']==20][trainData['taxi_code']== 1]),len(trainData[trainData['hour']==21][trainData['taxi_code']== 1]),len(trainData[trainData['hour']==22][trainData['taxi_code']== 1]),len(trainData[trainData['hour']==23][trainData['taxi_code']== 1]))
Base2 = (len(trainData[trainData['hour']==0][trainData['taxi_code']== 2]), len(trainData[trainData['hour']==1][trainData['taxi_code']== 2]), len(trainData[trainData['hour']==2][trainData['taxi_code']== 2]), len(trainData[trainData['hour']==3][trainData['taxi_code']== 2]),len(trainData[trainData['hour']==4][trainData['taxi_code']== 2]),len(trainData[trainData['hour']==5][trainData['taxi_code']== 2]),len(trainData[trainData['hour']==6][trainData['taxi_code']== 2]),len(trainData[trainData['hour']==7][trainData['taxi_code']== 2]),len(trainData[trainData['hour']==8][trainData['taxi_code']== 2]),len(trainData[trainData['hour']==9][trainData['taxi_code']== 2]),len(trainData[trainData['hour']==10][trainData['taxi_code']== 2]),len(trainData[trainData['hour']==11][trainData['taxi_code']== 2]),len(trainData[trainData['hour']==12][trainData['taxi_code']== 2]),len(trainData[trainData['hour']==13][trainData['taxi_code']== 2]), len(trainData[trainData['hour']==14][trainData['taxi_code']== 2]),len(trainData[trainData['hour']==15][trainData['taxi_code']== 2]),len(trainData[trainData['hour']==16][trainData['taxi_code']== 2]),len(trainData[trainData['hour']==17][trainData['taxi_code']== 2]),len(trainData[trainData['hour']==18][trainData['taxi_code']== 2]),len(trainData[trainData['hour']==19][trainData['taxi_code']== 2]),len(trainData[trainData['hour']==20][trainData['taxi_code']== 2]),len(trainData[trainData['hour']==21][trainData['taxi_code']== 2]),len(trainData[trainData['hour']==22][trainData['taxi_code']== 2]),len(trainData[trainData['hour']==23][trainData['taxi_code']== 2]))
Base3 = (len(trainData[trainData['hour']==0][trainData['taxi_code']== 3]), len(trainData[trainData['hour']==1][trainData['taxi_code']== 3]), len(trainData[trainData['hour']==2][trainData['taxi_code']== 3]), len(trainData[trainData['hour']==3][trainData['taxi_code']== 3]),len(trainData[trainData['hour']==4][trainData['taxi_code']== 3]),len(trainData[trainData['hour']==5][trainData['taxi_code']== 3]),len(trainData[trainData['hour']==6][trainData['taxi_code']== 3]),len(trainData[trainData['hour']==7][trainData['taxi_code']== 3]),len(trainData[trainData['hour']==8][trainData['taxi_code']== 3]),len(trainData[trainData['hour']==9][trainData['taxi_code']== 3]),len(trainData[trainData['hour']==10][trainData['taxi_code']== 3]),len(trainData[trainData['hour']==11][trainData['taxi_code']== 3]),len(trainData[trainData['hour']==12][trainData['taxi_code']== 3]),len(trainData[trainData['hour']==13][trainData['taxi_code']== 3]), len(trainData[trainData['hour']==14][trainData['taxi_code']== 3]),len(trainData[trainData['hour']==15][trainData['taxi_code']== 3]),len(trainData[trainData['hour']==16][trainData['taxi_code']== 3]),len(trainData[trainData['hour']==17][trainData['taxi_code']== 3]),len(trainData[trainData['hour']==18][trainData['taxi_code']== 3]),len(trainData[trainData['hour']==19][trainData['taxi_code']== 3]),len(trainData[trainData['hour']==20][trainData['taxi_code']== 3]),len(trainData[trainData['hour']==21][trainData['taxi_code']== 3]),len(trainData[trainData['hour']==22][trainData['taxi_code']== 3]),len(trainData[trainData['hour']==23][trainData['taxi_code']== 3]))
Base4 = (len(trainData[trainData['hour']==0][trainData['taxi_code']== 4]), len(trainData[trainData['hour']==1][trainData['taxi_code']== 4]), len(trainData[trainData['hour']==2][trainData['taxi_code']== 4]), len(trainData[trainData['hour']==3][trainData['taxi_code']== 4]),len(trainData[trainData['hour']==4][trainData['taxi_code']== 4]),len(trainData[trainData['hour']==5][trainData['taxi_code']== 4]),len(trainData[trainData['hour']==6][trainData['taxi_code']== 4]),len(trainData[trainData['hour']==7][trainData['taxi_code']== 4]),len(trainData[trainData['hour']==8][trainData['taxi_code']== 4]),len(trainData[trainData['hour']==9][trainData['taxi_code']== 4]),len(trainData[trainData['hour']==10][trainData['taxi_code']== 4]),len(trainData[trainData['hour']==11][trainData['taxi_code']== 4]),len(trainData[trainData['hour']==12][trainData['taxi_code']== 4]),len(trainData[trainData['hour']==13][trainData['taxi_code']== 4]), len(trainData[trainData['hour']==14][trainData['taxi_code']== 4]),len(trainData[trainData['hour']==15][trainData['taxi_code']== 4]),len(trainData[trainData['hour']==16][trainData['taxi_code']== 4]),len(trainData[trainData['hour']==17][trainData['taxi_code']== 4]),len(trainData[trainData['hour']==18][trainData['taxi_code']== 4]),len(trainData[trainData['hour']==19][trainData['taxi_code']== 4]),len(trainData[trainData['hour']==20][trainData['taxi_code']== 4]),len(trainData[trainData['hour']==21][trainData['taxi_code']== 4]),len(trainData[trainData['hour']==22][trainData['taxi_code']== 4]),len(trainData[trainData['hour']==23][trainData['taxi_code']== 4]))
Base5 = (len(trainData[trainData['hour']==0][trainData['taxi_code']== 5]), len(trainData[trainData['hour']==1][trainData['taxi_code']== 5]), len(trainData[trainData['hour']==2][trainData['taxi_code']== 5]), len(trainData[trainData['hour']==3][trainData['taxi_code']== 5]),len(trainData[trainData['hour']==4][trainData['taxi_code']== 5]),len(trainData[trainData['hour']==5][trainData['taxi_code']== 5]),len(trainData[trainData['hour']==6][trainData['taxi_code']== 5]),len(trainData[trainData['hour']==7][trainData['taxi_code']== 5]),len(trainData[trainData['hour']==8][trainData['taxi_code']== 5]),len(trainData[trainData['hour']==9][trainData['taxi_code']== 5]),len(trainData[trainData['hour']==10][trainData['taxi_code']== 5]),len(trainData[trainData['hour']==11][trainData['taxi_code']== 5]),len(trainData[trainData['hour']==12][trainData['taxi_code']== 5]),len(trainData[trainData['hour']==13][trainData['taxi_code']== 5]), len(trainData[trainData['hour']==14][trainData['taxi_code']== 5]),len(trainData[trainData['hour']==15][trainData['taxi_code']== 5]),len(trainData[trainData['hour']==16][trainData['taxi_code']== 5]),len(trainData[trainData['hour']==17][trainData['taxi_code']== 5]),len(trainData[trainData['hour']==18][trainData['taxi_code']== 5]),len(trainData[trainData['hour']==19][trainData['taxi_code']== 5]),len(trainData[trainData['hour']==20][trainData['taxi_code']== 5]),len(trainData[trainData['hour']==21][trainData['taxi_code']== 5]),len(trainData[trainData['hour']==22][trainData['taxi_code']== 5]),len(trainData[trainData['hour']==23][trainData['taxi_code']== 5]))
Base6 = (len(trainData[trainData['hour']==0][trainData['taxi_code']== 6]), len(trainData[trainData['hour']==1][trainData['taxi_code']== 6]), len(trainData[trainData['hour']==2][trainData['taxi_code']== 6]), len(trainData[trainData['hour']==3][trainData['taxi_code']== 6]),len(trainData[trainData['hour']==4][trainData['taxi_code']== 6]),len(trainData[trainData['hour']==5][trainData['taxi_code']== 6]),len(trainData[trainData['hour']==6][trainData['taxi_code']== 6]),len(trainData[trainData['hour']==7][trainData['taxi_code']== 6]),len(trainData[trainData['hour']==8][trainData['taxi_code']== 6]),len(trainData[trainData['hour']==9][trainData['taxi_code']== 6]),len(trainData[trainData['hour']==10][trainData['taxi_code']== 6]),len(trainData[trainData['hour']==11][trainData['taxi_code']== 6]),len(trainData[trainData['hour']==12][trainData['taxi_code']== 6]),len(trainData[trainData['hour']==13][trainData['taxi_code']== 6]), len(trainData[trainData['hour']==14][trainData['taxi_code']== 6]),len(trainData[trainData['hour']==15][trainData['taxi_code']== 6]),len(trainData[trainData['hour']==16][trainData['taxi_code']== 6]),len(trainData[trainData['hour']==17][trainData['taxi_code']== 6]),len(trainData[trainData['hour']==18][trainData['taxi_code']== 6]),len(trainData[trainData['hour']==19][trainData['taxi_code']== 6]),len(trainData[trainData['hour']==20][trainData['taxi_code']== 6]),len(trainData[trainData['hour']==21][trainData['taxi_code']== 6]),len(trainData[trainData['hour']==22][trainData['taxi_code']== 6]),len(trainData[trainData['hour']==23][trainData['taxi_code']== 6]))
Base7 = (len(trainData[trainData['hour']==0][trainData['taxi_code']== 7]), len(trainData[trainData['hour']==1][trainData['taxi_code']== 7]), len(trainData[trainData['hour']==2][trainData['taxi_code']== 7]), len(trainData[trainData['hour']==3][trainData['taxi_code']== 7]),len(trainData[trainData['hour']==4][trainData['taxi_code']== 7]),len(trainData[trainData['hour']==5][trainData['taxi_code']== 7]),len(trainData[trainData['hour']==6][trainData['taxi_code']== 7]),len(trainData[trainData['hour']==7][trainData['taxi_code']== 7]),len(trainData[trainData['hour']==8][trainData['taxi_code']== 7]),len(trainData[trainData['hour']==9][trainData['taxi_code']== 7]),len(trainData[trainData['hour']==10][trainData['taxi_code']== 7]),len(trainData[trainData['hour']==11][trainData['taxi_code']== 7]),len(trainData[trainData['hour']==12][trainData['taxi_code']== 7]),len(trainData[trainData['hour']==13][trainData['taxi_code']== 7]), len(trainData[trainData['hour']==14][trainData['taxi_code']== 7]),len(trainData[trainData['hour']==15][trainData['taxi_code']== 7]),len(trainData[trainData['hour']==16][trainData['taxi_code']== 7]),len(trainData[trainData['hour']==17][trainData['taxi_code']== 7]),len(trainData[trainData['hour']==18][trainData['taxi_code']== 7]),len(trainData[trainData['hour']==19][trainData['taxi_code']== 7]),len(trainData[trainData['hour']==20][trainData['taxi_code']== 7]),len(trainData[trainData['hour']==21][trainData['taxi_code']== 7]),len(trainData[trainData['hour']==22][trainData['taxi_code']== 7]),len(trainData[trainData['hour']==23][trainData['taxi_code']== 7]))

rects1 = plt.bar(index, Base1 , bar_width,
                 alpha=opacity,
                 color='black',label = 'B02512' )
rects2 = plt.bar(index+bar_width, Base2 , bar_width,
                 alpha=opacity,
                 color='red',label = 'B02598' )
rects3 = plt.bar(index+bar_width*2, Base3 , bar_width,
                 alpha=opacity,
                 color='blue',label = 'B02617' )
rects4 = plt.bar(index+bar_width*3, Base4 , bar_width,
                 alpha=opacity,
                 color='orange',label = 'B02682' )
rects5 = plt.bar(index+bar_width*4, Base5 , bar_width,
                 alpha=opacity,
                 color='green',label = 'B02764' )
#rects6 = plt.bar(index+bar_width*5, Base6 , bar_width,
#                 alpha=opacity,
#                 color='yellow',label = 'B02765' )
rects7 = plt.bar(index+bar_width*5, Base7 , bar_width,
                 alpha=opacity,
                 color='purple',label = 'Lyft' )
plt.xlabel('Hour')
plt.ylabel('Count for BaseVehicles ')
plt.title('Hourly Analysis - August')
plt.xticks(index + bar_width, ('0', '1', '2', '3','4','5','6','7','8','9','10', '11', '12', '13','14','15','16','17', '18', '19', '20','21','22','23'))
plt.legend(loc='top left')
 
plt.tight_layout()
plt.savefig('MazaMap.png',dpi=100)
plt.show()

#ENd of Plot Graph


#Accuracy calculation
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
#gnb = GaussianNB()
#X=trainData[["Lat", "Lon"]]
print("MultinomialNB..")
start_time=time.time()
model = GaussianNB()
model.fit(X,y)
prediction=model.predict(X_test)
print(accuracy_score(prediction,Y_test))
print(confusion_matrix(prediction,Y_test))
print(precision_score(Y_test,prediction, average=None))
print(recall_score(Y_test,prediction, average=None))
print("--- %s seconds ---" % (time.time() - start_time))
#print("KNN..")
print("GaussianNB..")
#print("SVM..")

print("KNN..")
start_time=time.time()
model = KNeighborsClassifier(n_neighbors=200, weights='uniform')
model.fit(X,y)
prediction=model.predict(X_test)
print(accuracy_score(prediction,Y_test))
print(confusion_matrix(prediction,Y_test))
print(precision_score(Y_test,prediction, average=None))
print(recall_score(Y_test,prediction, average=None))
print("--- %s seconds ---" % (time.time() - start_time))
print("Decision tree with AdaBoost..")


start_time = time.time()
print("With NN")
model = MLPClassifier(solver='adam',alpha=0.001,hidden_layer_sizes=(200),random_state=1,learning_rate_init=0.01)
model.fit(X,y)
#Prediction
prediction=model.predict(X_test)
print (accuracy_score(prediction, Y_test))
print(confusion_matrix(prediction,Y_test))
print(precision_score(Y_test,prediction, average=None))
print(recall_score(Y_test,prediction, average=None))
print("--- %s seconds ---" % (time.time() - start_time))
start_time = time.time()
#X=np.array(X)
#Y=np.array(trainData["taxi_code"])
start_time=time.time()
#model = KNeighborsClassifier(n_neighbors=5, weights='distance', radius=5)
#model = GaussianNB(priors=None)
#model = SVC(kernel='linear', probability=False, C=0.01)
model = AdaBoostClassifier(DecisionTreeClassifier(max_depth=100),n_estimators=100,learning_rate=0.05)
model.fit(X,y)
print(model)
#test=np.array(testData[["Lat","Lon"]])
prediction=model.predict(X_test)
print(accuracy_score(prediction,Y_test))
print(confusion_matrix(prediction,Y_test))
print(precision_score(Y_test,prediction, average=None))
print(recall_score(Y_test,prediction, average=None))
print("--- %s seconds ---" % (time.time() - start_time))

