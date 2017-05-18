from pandas import Series
import pandas as pd
import numpy as np
import os
import matplotlib.pylab as plt
from sklearn.cross_validation import train_test_split
from sklearn import tree
from sklearn.metrics import classification_report
import sklearn.metrics

os.chdir("E:\ComputerScience\Sem2\ML\Assignment1\Assignment1_DecisionTree\Files")
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


classifier = tree.DecisionTreeClassifier()
classifier = classifier.fit(pred_train, tar_train)

predictions = classifier.predict(pred_test)
#print(predictions.shape)

print("Classification report for classifier %s:\n%s\n"
      % (classifier, classification_report(tar_test, predictions)))

print("Confusion matrix:",  sklearn.metrics.confusion_matrix(tar_test,predictions))

#sklearn.metrics.confusion_matrix(tar_test,predictions)
print("Training Accuracy:")
print(sklearn.metrics.accuracy_score(tar_test, predictions))

#For Testing data
test_data = pd.read_csv("optdigits_test.csv")
test_data_clean = test_data.dropna()

X_train = data_clean[data_clean.columns[0:64]]
Y_train = data_clean[data_clean.columns[64:65]]

X_test = test_data_clean[test_data_clean.columns[0:64]]
Y_test = test_data_clean[test_data_clean.columns[64:65]]

#print(X_train.shape)
#print(X_test.shape)
#print(Y_train.shape)
#print(Y_test.shape)

new_classifier = tree.DecisionTreeClassifier()
new_classifier = new_classifier.fit(X_train, Y_train)
print("***************************************************************")
new_predictions = new_classifier.predict(X_test)
#print(new_predictions)
#print(Y_test)

sklearn.metrics.confusion_matrix(Y_test,new_predictions)
print("Testing Accuracy:")
print(sklearn.metrics.accuracy_score(Y_test, new_predictions))

from sklearn import tree
from io import StringIO
from IPython.display import Image
out = StringIO()
#tree.export_graphviz(classifier, out_file=out)
import pydotplus

print("Creating graph image:")
dot_data = tree.export_graphviz(classifier, out_file=out) 
graph=pydotplus.graph_from_dot_data(out.getvalue())
#print(out.getvalue())
Image(graph.create_png())
graph.write_png("OPTTree.png")

print("Execution completed.")





