from pandas import Series
import pandas as pd
import numpy as np
import os
import matplotlib.pylab as plt
from sklearn.cross_validation import train_test_split
from sklearn import svm
from sklearn.metrics import classification_report
import sklearn.metrics
from sklearn.model_selection import cross_val_score

print("Reading dataset.")
DR_data = pd.read_csv("optdigits_raining.csv")
data_clean = DR_data.dropna()
print(data_clean.shape)

predictors = data_clean[data_clean.columns[0:64]] 
#print(predictors)
targets = data_clean[data_clean.columns[64:65]] 

pred_train, pred_test, tar_train, tar_test = train_test_split(predictors, targets, test_size=.4)

#print(pred_train.shape)
#print(pred_test.shape)
##print(tar_train.shape)
#print(tar_test.shape)
print("Sample rbf")

classifier0 = svm.SVC(kernel="rbf")
classifier0 = classifier0.fit(pred_train, tar_train)

predictions0 = classifier0.predict(pred_test)
#print(predictions.shape)

sklearn.metrics.confusion_matrix(tar_test,predictions0)
print("Training Accuracy 1:")
trainscore0 = sklearn.metrics.accuracy_score(tar_test, predictions0)
print(trainscore0)
print("***************************************************************")

print("Sample 1")

classifier = svm.SVC(kernel="linear")
classifier = classifier.fit(pred_train, tar_train)

predictions = classifier.predict(pred_test)
#print(predictions.shape)

sklearn.metrics.confusion_matrix(tar_test,predictions)
print("Training Accuracy 1:")
trainscore = sklearn.metrics.accuracy_score(tar_test, predictions)
print(trainscore)
print("***************************************************************")


print("Sample 2")

classifier2 = svm.SVC(kernel="linear", C=10)
classifier2 = classifier2.fit(pred_train, tar_train)

predictions2 = classifier2.predict(pred_test)
#print(predictions.shape)

sklearn.metrics.confusion_matrix(tar_test,predictions2)
print("Training Accuracy 2:")
trainscore2 = sklearn.metrics.accuracy_score(tar_test, predictions2)
print(trainscore2)
print("***************************************************************")


print("Sample 3")

classifier3 = svm.SVC(kernel="linear", C=0.001)
classifier3 = classifier3.fit(pred_train, tar_train)

predictions3 = classifier3.predict(pred_test)
#print(predictions.shape)

sklearn.metrics.confusion_matrix(tar_test,predictions3)
print("Training Accuracy 3:")
trainscore3 = sklearn.metrics.accuracy_score(tar_test, predictions3)
print(trainscore3)
c, r = targets.shape
targets = targets.values.reshape(c,)

scores = cross_val_score(classifier3,predictors, targets, cv=5)
print("Scores: ", scores)
print("***************************************************************")


print("Sample 4")

classifier4 = svm.SVC(kernel="linear", C=0.00001)
classifier4 = classifier4.fit(pred_train, tar_train)

predictions4 = classifier4.predict(pred_test)
#print(predictions.shape)

sklearn.metrics.confusion_matrix(tar_test,predictions4)
print("Training Accuracy 4:")
trainscore4 = sklearn.metrics.accuracy_score(tar_test, predictions4)
print(trainscore4)

#For Testing data
test_data = pd.read_csv("optdigits_test.csv")
test_data_clean = test_data.dropna()

X_train = data_clean[data_clean.columns[0:64]]
Y_train = data_clean[data_clean.columns[64:65]]

X_test = test_data_clean[test_data_clean.columns[0:64]]
Y_test = test_data_clean[test_data_clean.columns[64:65]]

print("***************************************************************")

#print(X_train.shape)
#print(X_test.shape)
#print(Y_train.shape)
#print(Y_test.shape)

print("Test Sample 0 rbf kernel")

new_classifier0 = svm.SVC(kernel="rbf")
new_classifier0 = new_classifier0.fit(X_train, Y_train)

new_predictions0 = new_classifier0.predict(X_test)
sklearn.metrics.confusion_matrix(Y_test,new_predictions0)
print("Testing Accuracy 1:")
accuracy0 = sklearn.metrics.accuracy_score(Y_test, new_predictions0)
print(accuracy0)

print("***************************************************************")

print("Test Sample 1")

new_classifier = svm.SVC(kernel="linear")
new_classifier = new_classifier.fit(X_train, Y_train)

new_predictions = new_classifier.predict(X_test)
sklearn.metrics.confusion_matrix(Y_test,new_predictions)
print("Testing Accuracy 1:")
accuracy = sklearn.metrics.accuracy_score(Y_test, new_predictions)
print(accuracy)

print("***************************************************************")

print("Test Sample 2")

new_classifier2 = svm.SVC(kernel="linear", C=10)
new_classifier2 = new_classifier2.fit(X_train, Y_train)
new_predictions2 = new_classifier2.predict(X_test)

sklearn.metrics.confusion_matrix(Y_test,new_predictions2)
print("Testing Accuracy 2:")
accuracy2 = sklearn.metrics.accuracy_score(Y_test, new_predictions2)
print(accuracy2)


print("***************************************************************")

print("Test Sample 3")

new_classifier3 = svm.SVC(kernel="linear", C=0.001)
new_classifier3 = new_classifier3.fit(X_train, Y_train)
new_predictions3 = new_classifier3.predict(X_test)

sklearn.metrics.confusion_matrix(Y_test,new_predictions3)
print("Testing Accuracy 3:")
accuracy3 = sklearn.metrics.accuracy_score(Y_test, new_predictions3)
print(accuracy3)

print("***************************************************************")

print("Test Sample 4")

new_classifier4 = svm.SVC(kernel="linear", C=0.0001)
new_classifier4 = new_classifier4.fit(X_train, Y_train)
new_predictions4 = new_classifier4.predict(X_test)

sklearn.metrics.confusion_matrix(Y_test,new_predictions4)
print("Testing Accuracy 2:")
accuracy4 = sklearn.metrics.accuracy_score(Y_test, new_predictions4)
print(accuracy4)

import matplotlib.pyplot as plt
plt.plot([accuracy, accuracy2, accuracy3, accuracy4], color='red')
plt.plot([trainscore, trainscore2, trainscore3, trainscore4], color='blue')
plt.ylabel('Accuracy')
plt.show()

plt.plot( [0.0001,0.001,1,10], [trainscore4, trainscore3, trainscore, trainscore2], 'ro')
plt.xlim(0.0001, 11)
plt.ylim(0.9,1.0)
plt.xlabel('C value')
plt.ylabel('Accuracy')
plt.show()

yval = np.array([1,2,3,4,5])
plt.plot([yval],[scores], "ro")
plt.xlim(0, 6)
plt.show()








