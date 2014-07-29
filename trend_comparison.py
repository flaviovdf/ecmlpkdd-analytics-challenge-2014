#-*- coding: utf8
from __future__ import division, print_function

from gcv_ols import OLS

from sklearn import cluster
from sklearn import decomposition
from sklearn import preprocessing

from pyksc.dist import dist_all

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
    k = 50 
    
    print('K-means')
    D = transform_km(T12_train, k)
    X_train_new = np.hstack((D,  X_train))
    
    model = OLS()
    model.fit(X_train_new, Y_train)
    print(k, np.sqrt(model.G.mean(axis=0)))

    print('KSC')
    C = np.genfromtxt('ksc-results/cents_visits_%d.dat' % k, dtype='d')
    T_nolog = np.asarray(np.exp(T12_train) - 1, order='C')
    D = dist_all(C, T_nolog, rolling=True)[0].T
    X_train_new = np.hstack((D,  X_train))
    
    model = OLS()
    model.fit(X_train_new, Y_train)
    print(k, np.sqrt(model.G.mean(axis=0)))
