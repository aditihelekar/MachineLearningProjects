import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split
from sklearn.neural_network import MLPClassifier

print("Reading dataset.")
DR_data = pd.read_csv("optdigits_raining.csv")
data_clean = DR_data.dropna()
print(data_clean.shape)

predictors = data_clean[data_clean.columns[0:64]]
targets = data_clean[data_clean.columns[64:65]]
pred_train, pred_test, tar_train, tar_test = train_test_split(predictors, targets, test_size=.4)

classifier = MLPClassifier(hidden_layer_sizes=(35),solver='sgd',learning_rate_init=0.001,max_iter=1000)
classifier = classifier.fit(pred_train, tar_train)
predictions = classifier.predict(pred_test)
print("Training Accuracy: %f" % classifier.score(pred_test, tar_test))
print("Training set loss: %f" % classifier.loss_)




#For Testing data
test_data = pd.read_csv("optdigits_test.csv")
test_data_clean = test_data.dropna()

X_train = data_clean[data_clean.columns[0:64]]
Y_train = data_clean[data_clean.columns[64:65]]

X_test = test_data_clean[test_data_clean.columns[0:64]]
Y_test = test_data_clean[test_data_clean.columns[64:65]]

new_classifier = MLPClassifier(hidden_layer_sizes=(35),solver='sgd',learning_rate_init=0.001,max_iter=1000)
new_classifier = new_classifier.fit(X_train, Y_train)
print("***************************************************************")
new_predictions = new_classifier.predict(X_test)
print("Testing Accuracy: %f"  % new_classifier.score(X_test, Y_test))
print("Testing set loss: %f" % new_classifier.loss_)




plt.title("Training Loss")
plt.plot(classifier.loss_curve_)
plt.show()

plt.title("Testing Loss")
plt.plot(new_classifier.loss_curve_)
plt.show()

print("Execution completed.")




