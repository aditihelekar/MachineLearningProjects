import pandas as pd
import os
import numpy as np

os.chdir("E:\ComputerScience\Sem2\ML\Assignment1\Assignment1_DecisionTree\Files")
path = "amazon_baby_train.csv"
print("Reading dataset:")
sms = pd.read_table(path, header=None,sep=',', names=['name','review','rating'])
data_clean = sms.dropna()

# Add column to identify if rating is good= 1 and bad = 0
data_clean['rating_num'] = data_clean.rating.map({'1':0,'2':0,'3':1,'4':1,'5':1})
data_clean = data_clean[data_clean.rating_num != None]
data_clean = data_clean.drop(data_clean.index[[0]])
data_clean['rating_num'] = data_clean['rating_num'].apply(np.int64)


# how to define X and y for use with COUNTVECTORIZER
X = data_clean.review
y = data_clean.rating_num
print(X.shape)
print(y.shape)

# split X and y into training and testing sets
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.4)
#print(X_train.shape)
#print(X_test.shape)
#print(y_train.shape)
#print(y_test.shape)

# instantiate the vectorizer
from sklearn.feature_extraction.text import CountVectorizer
vect = CountVectorizer()
# learn training data vocabulary, then use it to create a document-term matrix
vect.fit(X_train)
X_train_dtm = vect.transform(X_train)
# equivalently: combine fit and transform into a single step
X_train_dtm = vect.fit_transform(X_train)
print("After fit transform")
X_test_dtm = vect.transform(X_test)
#print(X_test_dtm)

from sklearn import tree
classifier = tree.DecisionTreeClassifier()
# train the model using X_train_dtm (timing it with an IPython "magic command")
classifier = classifier.fit(X_train_dtm, y_train)

# make class predictions for X_test_dtm
y_pred_class = classifier.predict(X_test_dtm)
#print(y_pred_class)
#print(y_test)

import sklearn.metrics
# print the confusion matrix
sklearn.metrics.confusion_matrix(y_test.astype(int), y_pred_class)
print("Training Accuracy:")

print(sklearn.metrics.accuracy_score(y_test, y_pred_class))


####################################################################################
#For Testing data

testpath = "amazon_baby_test.csv"
test_data = pd.read_table(testpath, header=None,sep=',', names=['name','review','rating'])

test_data_clean = test_data.dropna()

# convert label to a numerical variable
test_data_clean['rating_num'] = test_data_clean.rating.map({'1':0,'2':0,'3':1,'4':1,'5':1})
test_data_clean = test_data_clean[test_data_clean.rating_num != None]
test_data_clean = test_data_clean.drop(test_data_clean.index[[0]])
test_data_clean['rating_num'] = test_data_clean['rating_num'].apply(np.int64)


XT_train = data_clean.review
YT_train = data_clean.rating_num

XT_test = test_data_clean.review
YT_test = test_data_clean.rating_num

#print(XT_train.shape)
#print(XT_test.shape)
#print(YT_train.shape)
#print(YT_test.shape)

newvect = CountVectorizer()
newvect.fit(XT_train)
XT_train_dtm = vect.transform(XT_train)
XT_train_dtm = vect.fit_transform(XT_train)
XT_test_dtm = vect.transform(XT_test)
                          

new_classifier = tree.DecisionTreeClassifier()
new_classifier = new_classifier.fit(XT_train_dtm, YT_train)
print("------------------------------------------------------------------------")
new_predictions = new_classifier.predict(XT_test_dtm)
#print(new_predictions)

sklearn.metrics.confusion_matrix(YT_test.astype(int), new_predictions)
print("Testing Accuracy:")
print(sklearn.metrics.accuracy_score(YT_test.astype(int), new_predictions))

print("Creating tree image:")
#from sklearn import tree
#from io import StringIO
#from IPython.display import Image
#out = StringIO()
#import pydotplus

#dot_data = tree.export_graphviz(classifier, out_file=out) 
#graph=pydotplus.graph_from_dot_data(out.getvalue())
#Image(graph.create_png())
#graph.write_png("AMZTree.png")

print("Execution completed.")

