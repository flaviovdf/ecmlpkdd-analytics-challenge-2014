#-*- coding: utf8
from __future__ import division, print_function

from gcv_ols import OLS

from sklearn import cluster

import myio
import numpy as np

if __name__ == '__main__':
    X_train, T12_train, _ = myio.read_features(test=False)
    Y_train = myio.read_response_train()
   
    X_test, T12_test, _ = myio.load_features(test=True)
    
    T = np.concatenate((T12_train, T12_test), axis=0)
    
    Ys = []
    for i in xrange(20):
        km = cluster.MiniBatchKMeans(n_clusters=50)
        km = km.fit(T)
        
        D_train = km.transform(T12_train)
        X_train = np.hstack((D_train,  X_train))
        
        D_test = km.transform(T12_test)
        X_test = np.hstack((D_test, X_test))
   
        model = OLS()
        model.fit(X_train, Y_train)

        Y = model.predict(X_test)
        Y = np.exp(Y) - 1
        Ys.append(Y)

    Ys = np.asarray(Ys).mean(axis=0)
    np.savetxt('pred.dat', Ys)
