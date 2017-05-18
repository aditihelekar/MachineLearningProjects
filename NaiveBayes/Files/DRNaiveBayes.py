import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB
import sklearn.model_selection
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score

import warnings
warnings.filterwarnings("ignore")

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

print("Sample 1 Multinomial :::")

classifier1 = MultinomialNB()
classifier1.fit(pred_train, tar_train)
predictions = classifier1.predict(pred_test)
#print(classifier1.predict(pred_test))
print("Accuracy: ",classifier1.score(pred_test, tar_test))
sklearn.metrics.confusion_matrix(tar_test,predictions)
print("Training Accuracy 1:")
trainscore = sklearn.metrics.accuracy_score(tar_test, predictions)
print(trainscore)

c, r = targets.shape
targets = targets.values.reshape(c,)

scores = cross_val_score(classifier1,predictors, targets, cv=10)
print("Scores: ", scores)
sumScore = 0
for i in scores:
    sumScore = sumScore + i
average = sumScore/10
print("Average: ", average)
print("***************************************************************")



print("Sample 2 Gaussian:::")
classifier2 = GaussianNB()
classifier2.fit(pred_train, tar_train)

print(classifier2.predict(pred_test))
print("Accuracy: ",classifier2.score(pred_test, tar_test))

print("Sample 3 Bernoulli:::")
classifier3 = BernoulliNB()
classifier3.fit(pred_train, tar_train)

print(classifier3.predict(pred_test))
print("Accuracy: ",classifier3.score(pred_test, tar_test))






print("Testing::::")


#For Testing data
test_data = pd.read_csv("optdigits_test.csv")
test_data_clean = test_data.dropna()

X_train = data_clean[data_clean.columns[0:64]]
Y_train = data_clean[data_clean.columns[64:65]]

X_test = test_data_clean[test_data_clean.columns[0:64]]
Y_test = test_data_clean[test_data_clean.columns[64:65]]

print("Sample 1 Multinomial :::")

new_classifier1 = MultinomialNB()
new_classifier1.fit(X_train, Y_train)

print(new_classifier1.predict(X_test))
print("Accuracy: ",new_classifier1.score(X_test, Y_test))


print("Sample 2 Gaussian:::")
new_classifier2 = GaussianNB()
new_classifier2.fit(X_train, Y_train)

print(new_classifier2.predict(X_test))
print("Accuracy: ",new_classifier2.score(X_test, Y_test))

print("Sample 3 Bernoulli:::")
new_classifier3 = BernoulliNB()
new_classifier3.fit(pred_train, tar_train)
print(new_classifier3.predict(pred_test))
print("Accuracy: ",new_classifier3.score(pred_test, tar_test))




