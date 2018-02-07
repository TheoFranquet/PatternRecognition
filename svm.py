from itertools import combinations, product
from sklearn.svm import SVC
import numpy as np
from scipy.stats import mode
from sklearn.base import BaseEstimator

class MultiClassSVM(BaseEstimator):
    def __init__(self, C = 1, kernel = 'rbf', gamma = 1, fit_type = '1v1', coef0 = 0):
        self.C = C
        self.kernel = kernel
        self.gamma = gamma
        self.fit_type = fit_type
        self.coef0 = coef0

    def fit(self, x, y):
        if self.fit_type == '1v1':
            self.fit1v1(x, y)
        else:
            self.fit1vN(x, y)
            
    def fit1v1(self, x, y):
        x = np.asarray(x)
        y = np.asarray(y)
        
        self.classes_ = np.unique(y)
        self.svms = {}
        
        working_dict = {}
        
        for label in self.classes_:
            working_dict[label] = x[y == label]
        
        for l1, l2 in combinations(self.classes_, r = 2):
            iter_x = np.concatenate((
                working_dict[l1],
                working_dict[l2]
            ))
            iter_y = [l1] * len(working_dict[l1]) + [l2] * len(working_dict[l2])
            self.svms[(l1, l2)] = self.fit_(iter_x, iter_y)
    
    def fit1vN(self, x, y):
        x = np.asarray(x)
        y = np.asarray(y)
        
        self.classes_ = np.unique(y)
        self.svms = {}
        
        for label in self.classes_:
            iter_labels = np.zeros(len(y))
            iter_labels[y == label] = 1
            
            self.svms[label] = self.fit_(x, iter_labels)

    def fit_(self, x, y):
        cls_svm = SVC(C = self.C, kernel = self.kernel, gamma = self.gamma)
        cls_svm.fit(x, y)
        return cls_svm

    def predict(self, x):
        if self.fit_type == '1v1':
            return self.predict1v1(x)
        else:
            return self.predict1vN(x)

    def predict1v1(self, x):
        predictions = np.zeros((len(self.svms), len(x)))

        for i, p in enumerate(self.svms.keys()):
            predictions[i] = self.svms[p].predict(x)

        return  mode(predictions, axis = 0)[0][0]

    def predict1vN(self, x):
        predictions = np.zeros((len(self.classes_), len(x)))

        for i, p in enumerate(self.classes_):
            predictions[i] = self.svms[p].decision_function(x)
                
        return self.classes_[np.argmax(predictions, axis = 0)]


    def score(self, X, y, sample_weight=None):
        return np.mean(self.predict(X) == y)

    def get_params(self, deep = False):
        return {
            'kernel': self.kernel,
            'C': self.C,
            'gamma': self.gamma,
            'fit_type': self.fit_type,
            'coef0': self.coef0
        }

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)

        return self