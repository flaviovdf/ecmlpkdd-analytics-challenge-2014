#-*- coding: utf8
from __future__ import division, print_function

from gcv_ols import OLS

from rbf import RidgeRBFModel

from sklearn import cross_validation
from sklearn import grid_search

import myio
import numpy as np

def rbf(X, Y):
    for sigma in [0.001, 0.01, 0.1, 1, 10, 100, 1000]:
        for num_dist in [10, 50, 100]:
            model = RidgeRBFModel(sigma=sigma, num_dists=num_dist)
            model.fit(X, Y)
    
            base = model.base_learner
            alphas = base.alphas
            best = base.alpha_
            
            best_idx = np.where(alphas == best)[0][0]
            cv_values = base.cv_values_[:, :, best_idx]
            
            print('sigma', 'num_dists', 'alpha', 'rmse')
            print(sigma, num_dist, best, np.sqrt(cv_values.mean(axis=0)))

if __name__ == '__main__':
    
    X_ml_visits = myio.read_features_ml(test=False, series='visits')
    X_ml_facebook = myio.read_features_ml(test=False, series='facebook')
    X_ml_twitter = myio.read_features_ml(test=False, series='twitter')

    X_news = myio.read_features_news(test=False)
    Y_train = myio.read_response_train()
    
    print('ML visits')
    model = OLS()
    model.fit(X_ml_visits, Y_train)
    print(np.sqrt(model.G.mean(axis=0)))
    print()

    print('ML facebook')
    model = OLS()
    model.fit(X_ml_facebook, Y_train)
    print(np.sqrt(model.G.mean(axis=0)))
    print()

    print('ML twitter')
    model = OLS()
    model.fit(X_ml_twitter, Y_train)
    print(np.sqrt(model.G.mean(axis=0)))
    print()

    print('News')
    model = OLS()
    model.fit(X_news, Y_train)
    print(np.sqrt(model.G.mean(axis=0)))
    print()

    print('RBF visits')
    rbf(X_ml_visits, Y_train)
    print()

    print('RBF facebook')
    rbf(X_ml_facebook, Y_train)
    print()

    print('ML twitter')
    rbf(X_ml_twitter, Y_train)
    print()


