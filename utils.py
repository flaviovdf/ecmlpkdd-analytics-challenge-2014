# -*- coding: utf8
'''Utilities to run code which minimized the MRSE'''

from __future__ import division, print_function

import numpy as np

def mrse_transform(X, y):
    '''
    Transforms X so that mean relative error regression models can fit. More
    formally this is making the indexes [i, j] of x equal to:: 
    
        X[i, j] / y[j]

    y is also converted to a ones array. Fitting linear regression models on
    such data will optimize for mean relative squared error.
    
    Parameters
    ----------
    X : array-like of shape = [n_samples, n_features]
        Input samples
    y : array of shape = [n_samples]
        Response variable

    Returns
    -------
    X_new : array-like of shape = [n_samples, n_features]
        Transformed X
    y_new : array of shape = [n_samples]
        Array of ones
    '''
    X = np.asanyarray(X, dtype='d')
    y = np.asanyarray(y, dtype='d')

    if y.ndim > 1:
        raise ShapeException('y must have a single dimension')

    if X.shape[0] != y.shape[0]:
        raise ShapeException('X and y must have the same number of rows!')

    X_new = (X.T / y).T
    y_new = np.ones(y.shape)

    return X_new, y_new


