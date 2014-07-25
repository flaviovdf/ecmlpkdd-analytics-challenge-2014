#-*- coding: utf8
from __future__ import print_function, division

from pyksc import ksc

import myio
import numpy as np

def cluster(T, num_clust=5):
    '''
    Runs the KSC algorithm on time series matrix T.

    Parameters
    ----------
    T : ndarray of shape (row, time series length)
        The time series to cluster
    num_clust : int
        Number of clusters to create
    '''
    T = np.asarray(T + 1e-20, order='C').copy()
    cents, assign, _, _ = ksc.inc_ksc(T, num_clust)
    return cents, assign

if __name__ == '__main__':
    T_train_visits = myio.read_48h_timeseries('visits').values[:, :12]

    for num_clust in [10, 30, 50, 70, 90, 110]:
        cents_visits, assign_visits = cluster(T_train_visits, num_clust)
        np.savetxt('cents_visits_%d.dat' % num_clust, cents_visits)
        np.savetxt('assign_visits_%d.dat' % num_clust, assign_visits)
