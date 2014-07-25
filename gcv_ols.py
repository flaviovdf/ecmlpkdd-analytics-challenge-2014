#-*- coding: utf8
'''
A scikit-learn API like ordinary least squares algorithm with the added 
general cross validation scores embedded already pre-computed.
'''
from __future__ import division, print_function

import numpy as np

class OLS(object):

    def __init__(self):

        self.C = None
        self.G = None

    def fit(self, X, Y):

        PI = np.linalg.pinv(X)
        H = np.dot(X, PI)
        
        C = np.dot(PI, Y)
        Yhat = np.dot(X, C)

        R = Yhat - Y
        Aux = (R.T / (1 - np.diag(H))).T
        G = np.power(Aux, 2)
        
        self.C = C
        self.G = G

    def predict(self, X):
        
        return np.dot(X, self.C)
