'''
Author: Kalina Jasinska
'''

from sklearn import datasets
from classifiers_students.naive_bayes import NaiveBayesGaussian
from utils.evaluate import test_model

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
mgnb = NaiveBayesGaussian()
mgnb.fit(X_iris_train, y_iris_train)

test_model(mgnb, X_iris_train, y_iris_train, False)
test_model(mgnb, X_iris_test, y_iris_test, False)

# accuracy:
# 0.966666666667
# accuracy:
# 0.933333333333

# Compare the performance of your implementation
# to GaussianNB from sklearn to verify correctness
from sklearn.naive_bayes import GaussianNB

gnb = GaussianNB()
gnb.fit(X_iris_train, y_iris_train)

test_model(gnb, X_iris_train, y_iris_train, False)
test_model(gnb, X_iris_test, y_iris_test, False)

# accuracy:
# 0.966666666667
# accuracy:
# 0.933333333333