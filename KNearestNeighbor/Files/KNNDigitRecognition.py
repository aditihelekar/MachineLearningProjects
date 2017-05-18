import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import sklearn.model_selection

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

neigh = KNeighborsClassifier(n_neighbors=3)
neigh.fit(pred_train, tar_train)

print(neigh.predict(pred_test))
print(neigh.score(pred_test, tar_test))

print("Sample 2:::")


neigh1 = KNeighborsClassifier(n_neighbors=50)
neigh1.fit(pred_train, tar_train)

print(neigh1.predict(pred_test))
print(neigh1.score(pred_test, tar_test))

print("Sample 3:::")


neigh2 = KNeighborsClassifier(n_neighbors=5, weights='distance')
neigh2.fit(pred_train, tar_train)

print(neigh2.predict(pred_test))
print(neigh2.score(pred_test, tar_test))

print("Sample 4:::")


neigh3 = KNeighborsClassifier(n_neighbors=5, weights='distance',algorithm='ball_tree',leaf_size=50)
neigh3.fit(pred_train, tar_train)

print(neigh3.predict(pred_test))
print(neigh3.score(pred_test, tar_test))


print("Sample 5:::")

# creating odd list of K for KNN
myList = list(range(1,50))

# subsetting just the odd ones
neighbors = filter(lambda x: x % 2 != 0, myList)

# empty list that will hold cv scores
cv_scores = []

# perform 10-fold cross validation
for k in neighbors:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = sklearn.model_selection.cross_val_score(knn, pred_train, tar_train, cv=10, scoring='accuracy')
    cv_scores.append(scores.mean())
    
# changing to misclassification error
MSE = [1 - x for x in cv_scores]

# determining best k
optimal_k = neighbors[MSE.index(min(MSE))]
print ("The optimal number of neighbors is %d" % optimal_k)

# plot misclassification error vs k
plt.plot(neighbors, MSE)
plt.xlabel('Number of Neighbors K')
plt.ylabel('Misclassification Error')
plt.show()


neigh4 = KNeighborsClassifier(n_neighbors=5, weights='distance',algorithm='ball_tree',leaf_size=50,p=1)
neigh4.fit(pred_train, tar_train)

print(neigh4.predict(pred_test))
print(neigh4.score(pred_test, tar_test))


print("Testing::::")


#For Testing data
test_data = pd.read_csv("optdigits_test.csv")
test_data_clean = test_data.dropna()

X_train = data_clean[data_clean.columns[0:64]]
Y_train = data_clean[data_clean.columns[64:65]]

X_test = test_data_clean[test_data_clean.columns[0:64]]
Y_test = test_data_clean[test_data_clean.columns[64:65]]

neightest = KNeighborsClassifier(n_neighbors=5, weights='distance',algorithm='ball_tree',leaf_size=50)
neightest.fit(X_train, Y_train)

print(neightest.predict(X_test))
print(neightest.score(X_test, Y_test))


