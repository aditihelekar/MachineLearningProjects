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
trainDataAug = pd.read_csv('uber-raw-data-aug14.csv', sep=',',header=0,keep_default_na=False)
trainDataSep = pd.read_csv('uber-raw-data-sep14.csv', sep=',',header=0,keep_default_na=False)
trainDataLyft = pd.read_csv('other-Lyft.csv', sep=',',header=0,keep_default_na=False)

trainDataAug = trainDataAug.dropna()
trainDataSep = trainDataSep.dropna()
trainDataLyft = trainDataLyft.dropna()
#trainDataLyft = trainDataLyft['Lat'].replace(to_replace="[a-zA-Z]", value='40.77398', regex=True, inplace=True)
#trainDataLyft = trainDataLyft['Lon'].replace(to_replace="[a-zA-Z]", value='-73.97848', regex=True, inplace=True)
print(trainDataAug.shape)
print(trainDataSep.shape)
print(trainDataLyft.shape)

#frames = [trainDataAug, trainDataSep, trainDataLyft]

#trainData = pd.concat(frames)
#trainData = trainDataLyft
trainData = trainDataAug.append(trainDataLyft).append(trainDataSep)
#trainData.append(trainDataSep)
#trainData.append(trainDataLyft)
trainData["taxi_code"] = trainData['Taxi Service'].map({'Uber':1,'Lyft':2})
trainData = trainData.dropna()


trainData['Date/Time'] = pd.to_datetime(trainData['Date/Time'], infer_datetime_format =True)
print(trainData['Date/Time'][1])
trainData['hour'] = trainData['Date/Time'].dt.hour
print(trainData['hour'][1])
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


print("Model is -")
print("Prediction is - ")

from sklearn.model_selection import train_test_split

X = trainData[["Lat", "Lon", "hour"]]
y = trainData[["taxi_code"]]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.6, random_state=42)


#Accuracy calculation
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
#gnb = GaussianNB()
#X=trainData[["Lat", "Lon"]]
#model = MultinomialNB(alpha=0.0001, fit_prior=True, class_prior=None)
#model.fit(X,trainData["Base"])
#print("KNN..")
#print("GaussianNB..")
#print("SVM..")
print("Decision tree with AdaBoost..")
#X=np.array(X)
#Y=np.array(trainData["taxi_code"])
start_time=time.time()
#model = KNeighborsClassifier(n_neighbors=5, weights='distance', radius=5)
#model = GaussianNB(priors=None)
#model = SVC(kernel='linear', probability=False, C=0.01)
model = AdaBoostClassifier(DecisionTreeClassifier(max_depth=100),n_estimators=100,learning_rate=0.05)
model.fit(X_train,y_train)
print(model)
#test=np.array(testData[["Lat","Lon"]])
prediction=model.predict(X_test)
print(accuracy_score(prediction,y_test))
print(confusion_matrix(prediction,y_test))
print(precision_score(y_test,prediction, average=None))
print(recall_score(y_test,prediction, average=None))
print("--- %s seconds ---" % (time.time() - start_time))

