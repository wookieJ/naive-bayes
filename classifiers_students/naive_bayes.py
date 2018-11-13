import numpy as np
import math
from scipy.stats import norm
from sklearn.base import BaseEstimator


class NaiveBayesNominal:
    def __init__(self):
        # todo - names of attributes - use them
        # self.classes_ = None
        self.model = dict()
        # self.y_prior = []
        self.y_yes = 0
        self.y_yes_prob = 0
        self.norm = 1

    def fit(self, X, y):
        for out in y:
            if out == 1:
                self.y_yes += 1
        self.y_yes_prob = self.y_yes / (len(y) * 1.0)

        for idr, row in enumerate(X):
            for idx, x in enumerate(row):
                if idx not in self.model:
                    self.model[idx] = [[0, 0], [0, 0], [0, 0]]
                self.model[idx][x][y[idr]] += 1

        for key in self.model:
            for i in range(3):
                self.model[key][i][1] /= self.y_yes * 1.0
                self.model[key][i][0] /= (len(y) - self.y_yes) * 1.0

    def predict_proba(self, X):
        result = []
        for row in X:
            flu_yes = 1
            flu_no = 1
            for idx, x in enumerate(row):
                flu_yes *= self.model[idx][x][1]
                flu_no *= self.model[idx][x][0]
            self.norm = self.y_yes_prob * flu_yes + (1 - self.y_yes_prob) * flu_no
            flu_yes *= self.y_yes_prob
            flu_yes /= self.norm
            flu_no *= (1 - self.y_yes_prob)
            flu_no /= self.norm
            result.append((flu_no, flu_yes))

        return result

    def predict(self, X):
        flu_prob = self.predict_proba(X)
        flu_pred = []
        for flu in flu_prob:
            if flu[0] > flu[1]:
                flu_pred.append(0)
            else:
                flu_pred.append(1)

        return np.array(flu_pred)

class NaiveBayesNumNom(BaseEstimator):
    def __init__(self):
        raise NotImplementedError

    def fit(self, X, y):
        raise NotImplementedError

    def predict_proba(self, X):
        raise NotImplementedError

    def predict(self, X):
        raise NotImplementedError


class NaiveBayesGaussian:
    def __init__(self, is_cat=None, m=0.0):
        self.is_cat = is_cat
        self.m = m
        self.model = dict()
        self.classes = set()

    def fit(self, X, yy):
        for y in yy:
            self.classes.add(y)

        for col in range(len(X[0])):
            columns_by_keys = dict()
            for idx, x in enumerate(X[:, col]):
                if yy[idx] not in columns_by_keys:
                    columns_by_keys[yy[idx]] = []
                columns_by_keys[yy[idx]].append(x)

                if col not in self.model:
                    self.model[col] = dict()

                for key in columns_by_keys:
                    mean = self.mean(columns_by_keys[key])
                    # mean = np.mean(columns_by_keys[key])
                    std = self.stdev(columns_by_keys[key])
                    # std = np.std(columns_by_keys[key])
                    self.model[col][key] = (mean, std)

    def predict_proba(self, X):
        result = []
        for row in X:
            # todo - reduce containers
            tuple = dict()
            for key in self.classes:
                proba = 1
                for idx, x in enumerate(row):
                    proba *= self.calculateProbability(x, self.model[idx][key][0], self.model[idx][key][1])
                    # proba *= norm(self.model[idx][key][0], self.model[idx][key][1]).pdf(x)
                tuple[key] = proba
            result.append(tuple)
        return result

    def mean(self, numbers):
        return sum(numbers)/float(len(numbers))

    def stdev(self, numbers):
        avg = self.mean(numbers)
        variance = sum([pow(x-avg,2) for x in numbers])/float(len(numbers)-1)
        return math.sqrt(variance)

    def calculateProbability(self, x, mean, stdev):
        exponent = math.exp(-(math.pow(x-mean,2)/(2*math.pow(stdev,2))))
        return (1 / (math.sqrt(2*math.pi) * stdev)) * exponent

    def predict(self, X):
        pred = self.predict_proba(X)
        result = []

        for row in pred:
            max = [0, row[0]]
            for idr in row:
                if row[idr] > max[1]:
                    max = [idr, row[idr]]
            result.append(max[0])

        return np.array(result)