#-*- coding: utf8
from __future__ import division, print_function

from gcv_ols import OLS

from sklearn import linear_model
from sklearn import cluster
from sklearn import cross_validation
from sklearn import decomposition
from sklearn import grid_search
from sklearn import preprocessing

import myio
import numpy as np

def transform_pca(T, num_clusters):
    
    Z = preprocessing.StandardScaler().fit_transform(T.T)
    pca = decomposition.PCA(num_clusters)
    T = pca.fit_transform(Z)
    D = pca.components_
    
    return D.T

def transform_ica(T, num_clusters):
    
    Z = preprocessing.StandardScaler().fit_transform(T.T)
    ica = decomposition.FastICA(num_clusters)
    T = ica.fit_transform(Z)
    D = ica.mixing_

    return D

def transform_km(T, num_clusters):

    Z = preprocessing.StandardScaler().fit_transform(T)
    km = cluster.MiniBatchKMeans(n_clusters=num_clusters)
    km = km.fit(Z)
    D = km.transform(Z)

    return D

if __name__ == '__main__':
    
    X_train, T12_train, hosts_train = myio.read_features(test=False)
    Y_train = myio.read_response_train()
    
    cv = cross_validation.StratifiedKFold(hosts_train, 2)
    for first_half, second_half in cv:
        X_train = X_train[first_half]
        Y_train = Y_train[first_half]
        T12_train = T12_train[first_half]
        break

    for k in [2, 4, 8, 16, 32, 64, 128]:
        D = transform_km(T12_train, k)
        X_train_new = np.hstack((D,  X_train))
    
        model = OLS()
        model.fit(X_train_new, Y_train)
        print(np.sqrt(model.G.mean(axis=0)))
