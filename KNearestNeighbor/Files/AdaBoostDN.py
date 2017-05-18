import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier

print("Reading dataset.")
DR_data = pd.read_csv("optdigits_raining.csv")
data_clean = DR_data.dropna()
print(data_clean.shape)

predictors = data_clean[data_clean.columns[0:64]]
targets = data_clean[data_clean.columns[64:65]]
pred_train, pred_test, tar_train, tar_test = train_test_split(predictors, targets, test_size=.4)

print(pred_train.shape)
print(tar_train.shape)
print(pred_test.shape)

print("Sample 1")

classifier = AdaBoostClassifier(DecisionTreeClassifier(max_depth=2),n_estimators=500,learning_rate=1)
classifier.fit(pred_train, tar_train)
predictions = classifier.predict(pred_test)
print(accuracy_score(tar_test, predictions))

print("Sample 2")
classifier1 = AdaBoostClassifier(DecisionTreeClassifier(max_depth=2),n_estimators=1000,learning_rate=1)
classifier1.fit(pred_train, tar_train)
predictions1 = classifier1.predict(pred_test)
print(accuracy_score(tar_test, predictions1))

print("Sample 3")
classifier2 = AdaBoostClassifier(DecisionTreeClassifier(max_depth=2),n_estimators=1000,learning_rate=2)
classifier2.fit(pred_train, tar_train)
predictions2 = classifier2.predict(pred_test)
print(accuracy_score(tar_test, predictions2))

print("Sample 4")
classifier3 = AdaBoostClassifier(DecisionTreeClassifier(max_depth=5),n_estimators=1000,learning_rate=2)
classifier3.fit(pred_train, tar_train)
predictions3 = classifier3.predict(pred_test)
print(accuracy_score(tar_test, predictions3))

print("Sample 5")
classifier4 = AdaBoostClassifier(DecisionTreeClassifier(max_depth=10),n_estimators=1500,learning_rate=2)
classifier4.fit(pred_train, tar_train)
predictions4 = classifier4.predict(pred_test)
print(accuracy_score(tar_test, predictions4))


print("Testing Data::")
#For Testing data
test_data = pd.read_csv("optdigits_test.csv")
test_data_clean = test_data.dropna()

X_train = data_clean[data_clean.columns[0:64]]
Y_train = data_clean[data_clean.columns[64:65]]

X_test = test_data_clean[test_data_clean.columns[0:64]]
Y_test = test_data_clean[test_data_clean.columns[64:65]]

print(X_train.shape)
print(X_test.shape)
print(Y_train.shape)
print(Y_test.shape)

new_classifier = AdaBoostClassifier(DecisionTreeClassifier(max_depth=2),n_estimators=1000,learning_rate=2)
new_classifier.fit(X_train, Y_train)
predictionsTest = new_classifier.predict(X_test)
print(accuracy_score(Y_test, predictionsTest))


new_classifier1 = AdaBoostClassifier(DecisionTreeClassifier(max_depth=5),n_estimators=1000,learning_rate=2)
new_classifier1.fit(X_train, Y_train)
predictionsTest1 = new_classifier1.predict(X_test)
print(accuracy_score(Y_test, predictionsTest1))

new_classifier2 = AdaBoostClassifier(DecisionTreeClassifier(max_depth=10),n_estimators=1500,learning_rate=2)
new_classifier2.fit(X_train, Y_train)
predictionsTest2 = new_classifier2.predict(X_test)
print(accuracy_score(Y_test, predictionsTest2))

