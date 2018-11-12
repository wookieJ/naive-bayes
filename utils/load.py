'''
Author: Kalina Jasinska
'''

import pandas
import numpy as np
from sklearn.preprocessing import OneHotEncoder

def convert_flu(data, data_test):
    for m in [data, data_test]:
        for ftr in ['katar', 'goraczka', 'dreszcze', 'grypa']:
            if ftr in m:
                m[ftr] = m[ftr].map({'tak': 1, 'nie': 0, 1: 1, 0: 0})

        for ftr in ['bol_glowy']:
            if ftr in m:
                m[ftr] = m[ftr].map({'duzy': 2, 'sredni': 1, 'nie': 0, 2: 2, 1: 1, 0: 0})

    return data, data_test


def load_flu():
    data_file = "../data/grypa1-train.csv"
    test_file = "../data/grypa1-test.csv"

    data = pandas.read_csv(data_file)
    data_test = pandas.read_csv(test_file)

    data, data_test = convert_flu(data, data_test)

    evidence_cols = ['dreszcze', 'katar', 'bol_glowy', 'goraczka']
    Xpandas = data[evidence_cols]
    ypandas = data['grypa']

    X = Xpandas.values
    y = ypandas.values
    X_test = data_test.values

    return X, y, X_test


def is_known_binary(ftr_valid_values):
    if len(ftr_valid_values) == 2:
        if 'f' in ftr_valid_values and 't' in ftr_valid_values:
            return True
        if "+" in ftr_valid_values and "-" in ftr_valid_values:
            return True
    return False


def convert_to_numerical(data):
    binary_features = {"t": 1, "f": 0, "+": 1, "-": 0, "tak": 1, "nie": 0}
    # if ? in data: keep ? and then convert to sparse dataset to forget about "?"s
    is_categorical = []
    contains_missing = False

    ftr_index = 0
    for ftr in data.columns:
        if (data.dtypes[ftr_index] not in [np.float64, np.int64]) and "numeric" not in ftr:

            ftr_valid_values = ftr.split("_")[-1].split(".")
            ftr_uniq = data[ftr].unique()
            if "?" in ftr_uniq:
                contains_missing = True

            # if len(ftr_valid_values) == 2:
            #    is_categorical.append(False)
            # else:
            is_categorical.append(True)

            if is_known_binary(ftr_valid_values):
                orig_to_num = binary_features
            else:
                orig_to_num = dict(zip(ftr_valid_values, range(len(ftr_valid_values))))

            orig_to_num['?'] = np.nan
            data[ftr] = data[ftr].map(orig_to_num)

        else:
            is_categorical.append(False)
            # print "ftr is numeric: {0}".format(ftr)
        ftr_index += 1
    data = data.apply(pandas.to_numeric, errors='coerce')

    if contains_missing:
        data = pandas.SparseDataFrame(data)

    return data, contains_missing, is_categorical


def convert_to_numerical2(data1, data2):
    binary_features = {"t": 1, "f": 0, "+": 1, "-": 0}

    contains_missing = False
    is_categorical = []

    ftr_index = 0
    for ftr in data1.columns:
        if (data1.dtypes[ftr_index] not in [np.float64, np.int64]) and ("numeric" not in ftr):

            # print "Convert {0} {1}".format(ftr, data1.dtypes[ftr_index])
            ftr_valid_values = ftr.split("_")[-1].split(".")

            ftr_uniq1 = data1[ftr].unique()
            ftr_uniq2 = data2[ftr].unique()

            if "?" in ftr_uniq1 or "?" in ftr_uniq2:
                contains_missing = True

            # if len(ftr_valid_values) == 2:
            #    is_categorical.append(False)
            # else:
            is_categorical.append(True)

            if is_known_binary(ftr_valid_values):
                orig_to_num = binary_features
            else:
                orig_to_num = dict(zip(ftr_valid_values, range(len(ftr_valid_values))))
                # print ftr_valid_values
            orig_to_num['?'] = np.nan
            # print orig_to_num
            data1[ftr] = data1[ftr].map(orig_to_num)
            data2[ftr] = data2[ftr].map(orig_to_num)
        else:
            is_categorical.append(False)
            # print "Not convert  {0} {1}".format(ftr, data1.dtypes[ftr_index])
        ftr_index += 1

    data1 = data1.apply(pandas.to_numeric, errors='coerce')
    data2 = data2.apply(pandas.to_numeric, errors='coerce')

    if contains_missing:
        data1 = pandas.SparseDataFrame(data1)
        data2 = pandas.SparseDataFrame(data2)

    return data1, data2, contains_missing, is_categorical


def read_and_convert_pandas(fn):
    data = pandas.read_csv(fn)
    data, is_sparse, is_categorical = convert_to_numerical(data)

    ftrs = data.columns[0:-1].ravel()
    clas = [data.columns[-1]]

    if is_sparse:
        X = data[ftrs].to_coo().tocsr()
    else:
        X = data[ftrs].as_matrix()
    y = data[clas].as_matrix().ravel()

    return X, y, is_categorical


def read_and_convert_pandas_files(fn1, fn2):
    data1 = pandas.read_csv(fn1)
    data2 = pandas.read_csv(fn2)
    data1, data2, is_sparse, is_categorical = convert_to_numerical2(data1, data2)

    ftrs = data1.columns[0:-1].ravel()
    clas = [data1.columns[-1]]

    if is_sparse:
        X1 = data1[ftrs].to_coo().tocsr()
        X2 = data2[ftrs].to_coo().tocsr()
    else:
        X1 = data1[ftrs].as_matrix()
        X2 = data2[ftrs].as_matrix()

    y1 = data1[clas].as_matrix().ravel()
    y2 = data2[clas].as_matrix().ravel()

    return X1, y1, X2, y2, is_categorical


def convert_to_onehot(X, is_categorical):
    enc = OneHotEncoder(categorical_features=is_categorical, sparse=False)
    enc.fit(X)
    return enc.transform(X)  # .toarray()