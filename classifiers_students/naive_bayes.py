import numpy as np
import math
from scipy.stats import norm
from sklearn.base import BaseEstimator


class NaiveBayesNominal:
    def __init__(self):
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
                if x >= 1:
                    if idx not in self.model:
                        self.model[idx] = [0,0]
                    self.model[idx][y[idr]] += 1

        for key in self.model:
            self.model[key][1] /= self.y_yes * 1.0
            self.model[key][0] /= (len(y) - self.y_yes) * 1.0

    def predict_proba(self, X):
        result = []
        for idr, row in enumerate(X):
            flu_yes = 1
            flu_no = 1
            for idx, x in enumerate(row):
                if x >= 1:
                    flu_yes *= self.model[idx][1]
                    flu_no *= self.model[idx][0]
                else:
                    flu_yes *= (1 - self.model[idx][1])
                    flu_no *= (1 - self.model[idx][0])
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
            flu_pred.append(0) if flu[0] > flu[1] else flu_pred.append(1)

        return np.array(flu_pred)

class NaiveBayesGaussian:
    def __init__(self):
        raise NotImplementedError

    def fit(self, X, y):
        raise NotImplementedError

    def predict_proba(self, X):
        raise NotImplementedError

    def predict(self, X):
        raise NotImplementedError


class NaiveBayesNumNom(BaseEstimator):
    def __init__(self, is_cat=None, m=0.0):
        raise NotImplementedError

    def fit(self, X, yy):
        raise NotImplementedError

    def predict_proba(self, X):
        raise NotImplementedError

    def predict(self, X):
        raise NotImplementedError