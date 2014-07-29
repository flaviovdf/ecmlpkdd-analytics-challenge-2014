#-*- coding: utf8
from __future__ import division, print_function

from gcv_ols import OLS

from sklearn import cross_validation
from sklearn import grid_search

import myio
import numpy as np

if __name__ == '__main__':
    
    X_ml_visits = myio.read_features_ml(test=False, series='visits')
    X_ml_facebook = myio.read_features_ml(test=False, series='facebook')
    X_ml_twitter = myio.read_features_ml(test=False, series='twitter')

    X_news = myio.read_features(test=False)
    
    Y_train = myio.read_response_train()
    
    print('ML visits')
    model = OLS()
    model.fit(X_ml_visits, Y_train)
    print(np.sqrt(model.G.mean(axis=0)))

    print('ML facebook')
    model = OLS()
    model.fit(X_ml_facebook, Y_train)
    print(np.sqrt(model.G.mean(axis=0)))

    print('ML twitter')
    model = OLS()
    model.fit(X_ml_twitter, Y_train)
    print(np.sqrt(model.G.mean(axis=0)))

    print('News')
    model = OLS()
    model.fit(X_news, Y_train)
    print(np.sqrt(model.G.mean(axis=0)))
