import numpy as np
import pandas
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier

# Load and convert categorical data:
# sklearn does not directly support categorical data:
# http://scikit-learn.org/stable/modules/preprocessing.html#encoding-categorical-features

# Generally, scikit-learn works on any numeric data stored as numpy arrays
# or scipy sparse matrices. Other types that are convertible to numeric
# arrays such as pandas DataFrame are also acceptable.
# (http://scikit-learn.org/stable/faq.html)
from sklearn.tree import DecisionTreeClassifier


def convert_flu(data, data_test):
    for m in [data, data_test]:
        for ftr in ['katar', 'goraczka', 'dreszcze', 'grypa']:
            if ftr in m:
                m[ftr] = m[ftr].map({'tak': 1, 'nie': 0, 1: 1, 0: 0})

        for ftr in ['bol_glowy']:
            if ftr in m:
                m[ftr] = m[ftr].map({'duzy': 2, 'sredni': 1, 'nie': 0, 2: 2, 1: 1, 0: 0})

    return data, data_test

def test_model(m, X, y=None):
    ypred = m.predict(X)
    if y is not None:
        print "ground truth:"
        print y
    print "predicted:"
    print ypred
    if y is not None:
        print "accuracy:"
        print accuracy_score(y, ypred)

# Read data
data_file = "../data/grypa1-train.csv"
test_file = "../data/grypa1-test.csv"

data = pandas.read_csv(data_file)
data_test = pandas.read_csv(test_file)

# Print it
print data
print data_test
print data.columns

# Convert by encoding categories with indices
data, data_test = convert_flu(data, data_test)
print data
print data_test

# Divide the data into features and decision/class
evidence_cols = ['dreszcze', 'katar', 'bol_glowy', 'goraczka']
X = data[evidence_cols]
y = data['grypa']

# Convert to numpy matrices to use with sklearn
X_train  = X.as_matrix()
y_train = y.as_matrix()
X_test = data_test.as_matrix()

# Create a simple Neares Neighbour classifier:
knn = KNeighborsClassifier(n_neighbors=3)
# knn = DecisionTreeClassifier()
knn.fit(X_train, y_train)

# Evaluate the model on train data and check it's prediction on test example
print "train performance:"
test_model(knn, X_train, y_train)
print "test performance:"
test_model(knn, X_test)


#   dreszcze katar bol_glowy goraczka grypa
# 0      tak   nie    sredni      tak   nie
# 1      tak   tak       nie      nie   tak
# 2      tak   nie      duzy      tak   tak
# 3      nie   tak    sredni      tak   tak
# 4      nie   nie       nie      nie   nie
# 5      nie   tak      duzy      tak   tak
# 6      nie   tak      duzy      nie   nie
# 7      tak   tak    sredni      tak   tak
#   dreszcze katar bol_glowy goraczka
# 0      nie   tak       nie      tak
# Index([u'dreszcze', u'katar', u'bol_glowy', u'goraczka', u'grypa'], dtype='object')
#    dreszcze  katar  bol_glowy  goraczka  grypa
# 0         1      0          1         1      0
# 1         1      1          0         0      1
# 2         1      0          2         1      1
# 3         0      1          1         1      1
# 4         0      0          0         0      0
# 5         0      1          2         1      1
# 6         0      1          2         0      0
# 7         1      1          1         1      1
#    dreszcze  katar  bol_glowy  goraczka
# 0         0      1          0         1
# train performance:
# ground truth:
# [0 1 1 1 0 1 0 1]
# predicted:
# [1 1 1 1 0 1 1 1]
# accuracy:
# 0.75
# test performance:
# predicted:
# [1]