# -*- coding: utf8
from __future__ import division, print_function
'''
Implements the RBF model.
'''

import numpy as np

from scipy.spatial.distance import cdist

from sklearn.base import BaseEstimator
from sklearn.base import RegressorMixin

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import RidgeCV

class RidgeRBFModel(BaseEstimator, RegressorMixin):
    
    '''
    Implements rbf model by Pinto et al. 2013.

    Parameters
    ----------
    num_dists : integer
        number of distances to consider
    sigma : float
        smoothing in the rbf
    base_learner : None
        the base learner to use (any scikit learn regressor)
    '''

    def __init__(self, num_dists=2, sigma=0.1, base_learner=None, **kwargs):
        self.num_dists = num_dists
        self.sigma = sigma
        
        if base_learner is None:
            base_learner = RidgeCV(fit_intercept=False, \
                    alphas=[0.001, 0.01, 0.1, 100, 1000], cv=None,
                    store_cv_values=True)
        
        if 'fit_intercept' not in kwargs:
            kwargs['fit_intercept'] = False

        self.base_learner = base_learner.set_params(**kwargs)
        self.R = None
        self.model = None

    def fit(self, X, y):
        X = np.asanyarray(X, dtype='d')
        y = np.asanyarray(y, dtype='d')
        
        n = X.shape[0]
        num_dists = self.num_dists
        
        if self.num_dists > n:
            raise ParameterException('Number of distances is greater than ' + \
                    'num rows in X')

        if self.num_dists <= 0:
            self.R = None
        else:
            rand_idx = np.random.choice(X.shape[0], int(num_dists), replace=False)
            self.R = X[rand_idx]
            
            D = np.exp(-1.0 * ((cdist(X, self.R) ** 2) / (2 * (self.sigma ** 2))))
            X = np.hstack((X, D))

        #Un-comment for mrse code
        #X, y = mrse_transform(X, y)

        self.model = self.base_learner.fit(X, y)
        return self

    def predict(self, X):
        X = np.asanyarray(X, dtype='d')

        if self.R is not None:
            D = np.exp(-1.0 * ((cdist(X, self.R) ** 2) / (2 * (self.sigma ** 2))))
            X = np.hstack((X, D))

        return self.model.predict(X)

    def get_params(self, deep=True):
        rv = super(RidgeRBFModel, self).get_params(deep)
        bpm = self.base_learner.get_params()
        for name in self.base_learner.get_params():
            rv[name] = bpm[name]
        return rv
