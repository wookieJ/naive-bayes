'''
Author: Kalina Jasinska
'''

from sklearn import datasets
from utils.load import load_flu
from classifiers_students.naive_bayes import NaiveBayesNumNom
from utils.evaluate import test_model


print "Nominal dataset"

# Create a Naive Bayes Classifier supporting both numerical
# and categorical features

# Load categorical data using code from the previous exercise
X_flu_train, y_flu_train, X_flu_test = load_flu()

# Implement and test Naive Bayes Classifier supporting categorical data
nb = NaiveBayesNumNom()
nb.fit(X_flu_train, y_flu_train)

# Use the previously implemented testing function
# to evaluate the performance of the model
test_model(nb, X_flu_train, y_flu_train)
test_model(nb, X_flu_test)

# ground truth:
# [0 1 1 1 0 1 0 1]
# predicted:
# [1 1 1 1 0 1 0 1]
# accuracy:
# 0.875
# predicted:
# [1]

############################################################################################################################################################

print "Numerical dataset"

# Load some data provided with the sklearn package http://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_iris.html
iris = datasets.load_iris()
X_iris = iris.data
y_iris = iris.target

# Divide the dataset into training and test part 80:20 respectively
frac_train = 0.8
n_samples = X_iris.shape[0]
n_train = int(float(n_samples*frac_train))
X_iris_train = X_iris[:n_train, :]
y_iris_train = y_iris[:n_train]
X_iris_test = X_iris[n_train:, :]
y_iris_test = y_iris[n_train:]

# Implement and test Naive Bayes Classifier supporting categorical data
mgnb = NaiveBayesNumNom()
mgnb.fit(X_iris_train, y_iris_train)

test_model(mgnb, X_iris_train, y_iris_train, False)
test_model(mgnb, X_iris_test, y_iris_test, False)

# accuracy:
# 0.966666666667
# accuracy:
# 0.933333333333
