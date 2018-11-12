'''
Author: Kalina Jasinska
'''

from utils.load import load_flu
from classifiers_students.naive_bayes import NaiveBayesNominal
from utils.evaluate import test_model

# Load categorical data using code from the previous exercise
X_flu_train, y_flu_train, X_flu_test = load_flu()

# Implement and test Naive Bayes Classifier supporting categorical data
nb = NaiveBayesNominal()
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
