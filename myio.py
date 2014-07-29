#-*- coding: utf8
'''
IO code for prediction tasks. Here we have the methods to convert the
json code csv, which is faster to load / process using pandas.
'''
from __future__ import division, print_function

from config import DATAFRAME_FOLDER
from config import TRAIN_FOLDER
from config import TEST_FOLDER

from collections import OrderedDict

from sklearn.preprocessing import LabelBinarizer

import glob
import json
import os
import numpy as np
import pandas as pd

_48H_SERIES_NAMES = '%s_48h_series_train.csv'
TRAIN_CSV_FPATH = 'data_train.csv'
TEST_CSV_FPATH = 'data_test.csv'

def read_features_news(test=False):
    '''
    Loads the features used for the news model

    Parameters
    ----------
    test : bool
        Indicates if the test should be loaded instead of the train

    Returns
    -------
    X : pandas dataframe of features ; shape = (pages, n_features)
    '''
    df = read_csv(test)
    
    pts_start = 0
    pts_end = 12
    pts = range(pts_start, pts_end)

    idx_visits = ['series_visits_%d' % x for x in pts]
    idx_twitter = ['series_twitter_%d' % x for x in pts]
    idx_facebook = ['series_facebook_%d' % x for x in pts]

    #Captures the growth of each time series
    vals_visits = np.asarray([df[idx_visits].values.sum(axis=1)]).T
    vals_twitter = np.asarray([df[idx_twitter].values.sum(axis=1)]).T
    vals_facebook = np.asarray([df[idx_facebook].values.sum(axis=1)]).T
    
    #Log the values
    vals_visits = np.log(1 + vals_visits)
    vals_twitter = np.log(1 + vals_twitter)
    vals_facebook = np.log(1 + vals_facebook)

    #The features used for regression
    X = np.hstack((

        vals_visits,
        vals_twitter,
        vals_facebook,
   
        vals_visits * vals_twitter,
        vals_visits * vals_facebook,
        vals_twitter * vals_facebook,
        
        vals_visits ** 2,
        vals_twitter ** 2,
        vals_facebook ** 2,
        ))
    
    return X

def read_features_ml(test=False, series='visits'):
    '''
    Loads the features used for the ML model.

    Parameters
    ----------
    test : bool
        Indicates if the test should be loaded instead of the train

    series : str
        The time series to use

    Returns
    -------
    X : pandas dataframe of features ; shape = (pages, n_features)
    '''
    df = read_csv(test)
    
    pts_start = 0
    pts_end = 12
    pts = range(pts_start, pts_end)
    idx = ['series_%s_%d' % (series, x) for x in pts]
    
    X = np.log(1 + df[idx].values.cumsum(axis=1))

    return X

def read_features(test=False):
    '''
    Loads the features used for regression and the 12 points visits
    time series.

    Parameters
    ----------
    test : bool
        Indicates if the test should be loaded instead of the train

    Returns
    -------
    X : pandas dataframe of features ; shape = (pages, n_features)
    T : pandas dataframe of time series ; shape = (pages, 12)
    hosts : a ndarray for the ids of the hosts. Useful for cross \
            validation
    '''
    df = read_csv(test)
    
    pts_start = 0
    pts_end = 12
    pts = range(pts_start, pts_end)

    idx_visits = ['series_visits_%d' % x for x in pts]
    idx_twitter = ['series_twitter_%d' % x for x in pts]
    idx_facebook = ['series_facebook_%d' % x for x in pts]
    idx_active = ['series_average_active_time_%d' % x for x in pts]

    #Captures the growth of each time series
    vals_visits =  df[idx_visits].values.cumsum(axis=1)
    vals_twitter = df[idx_twitter].values.cumsum(axis=1)
    vals_facebook = df[idx_facebook].values.cumsum(axis=1)
    vals_active = df[idx_active].values.cumsum(axis=1)
    
    #Log the values
    vals_visits = np.log(1 + vals_visits)
    vals_twitter = np.log(1 + vals_twitter)
    vals_facebook = np.log(1 + vals_facebook)
    vals_active = np.log(1 + vals_active)

    hosts = df.values[:, 0]
    posted_h = df.values[:, 1]
    posted_d = df.values[:, 2]
    host_ids = np.unique(hosts)
    
    #Indicator variables for hosts, posted day and posted hours
    binarized_hosts = LabelBinarizer().fit_transform(hosts)
    binarized_posted_h = LabelBinarizer().fit_transform(posted_h)
    binarized_posted_d = LabelBinarizer().fit_transform(posted_d)

    #The original visits time series used for clustering
    T = np.log(1 + df[idx_visits].values)

    #The features used for regression
    X = np.hstack((

        binarized_hosts,
        binarized_posted_h,
        binarized_posted_d,

        vals_visits,
        vals_twitter,
        vals_facebook,
        vals_active,
   
        vals_visits * vals_twitter,
        vals_visits * vals_facebook,
        vals_twitter * vals_facebook,
        vals_active * vals_visits,
        vals_active * vals_twitter,
        vals_active * vals_facebook,
        
        vals_visits ** 2,
        vals_twitter ** 2,
        vals_facebook ** 2,
        vals_active ** 2,
        
        vals_visits ** 3,
        vals_twitter ** 3,
        vals_facebook ** 3,
        vals_active ** 3, 
        ))
    
    return X, T, hosts

def read_csv(test=False):
    '''
    Reads csv styled data time series as a pandas dataframe. Each row is a
    host and each column a feature from the original data.
    
    Parameters
    ----------
    test : bool
        Indicates if the test should be loaded instead of the train
    '''
    
    if test:
        fname = os.path.join(DATAFRAME_FOLDER, TEST_CSV_FPATH)
    else:
        fname = os.path.join(DATAFRAME_FOLDER, TRAIN_CSV_FPATH)
    
    return pd.DataFrame.from_csv(fname)

def read_response_train():
    '''
    Reads a matrix with the responses to be learnt. Each column
    of the matrix is the sum of 48h of visits, facebook likes or
    tweets. The rows are pages.
    '''

    #Create the response matrix
    T_train_visits = read_48h_timeseries('visits').values
    T_train_facebook = read_48h_timeseries('facebook').values
    T_train_twitter = read_48h_timeseries('twitter').values

    y_train_visits = np.log(1 +  T_train_visits.sum(axis=1))
    y_train_twitter = np.log(1 + T_train_twitter.sum(axis=1))
    y_train_facebook = np.log(1 + T_train_facebook.sum(axis=1))
    
    Y_train = np.vstack((y_train_visits, y_train_twitter, y_train_facebook)).T
    
    return Y_train

def read_48h_timeseries(series_name='visits'):
    '''
    Reads the 48h time series as a pandas dataframe. Each row is a
    host and each column a time tick.
    
    Parameters
    ----------
    test : bool
        Indicates if the test should be loaded instead of the train
    '''
    
    fname = os.path.join(DATAFRAME_FOLDER, _48H_SERIES_NAMES % series_name)
    return pd.DataFrame.from_csv(fname)

def to_48h_tseries_mat(series_name='visits'):
    '''
    Converts the json files to a pandas for the 48 hour time series.
    This is only available for the train dataset.

    Parameters
    ----------
    series_name : str
        Indicates which series to use, it can be 'visits', 'facebook',
        'twitter'.
    '''
    input_glob = os.path.join(TRAIN_FOLDER, '*.json')
    
    jsons = {}
    for f in glob.glob(input_glob):
        f_name = os.path.basename(f)
        host = f_name.split('.')[0]
        
        with open(f) as json_file:
            jsons[int(host)] = json.load(json_file)
    
    pages = OrderedDict()
    for host in sorted(jsons):
        for page in xrange(len(jsons[host]['pages'])): 
            page_vals = OrderedDict()
            pages[jsons[host]['pages'][page]['page_id']] = page_vals
            
            for page_attr in sorted(jsons[host]['pages'][page]):
                if 'series_48h' == page_attr:
                    series_dict = jsons[host]['pages'][page][page_attr]

                    n_ticks = len(series_dict[series_name])
                    for t in xrange(n_ticks):
                        page_vals['series_%s_%d' % (series_name, t)] = \
                                series_dict[series_name][t]
    
    return pd.DataFrame.from_dict(pages, orient='index')
 
def to_pandas(test=False):
    '''
    Converts the json files to a pandas dataframe. Each row of the
    dataframe is a webpage. Columns representfeatures

    Parameters
    ----------
    test : bool
        Indicates if the test should be loaded instead of the train
    '''
    if test:
        input_glob = os.path.join(TEST_FOLDER, '*.json')
    else:
        input_glob = os.path.join(TRAIN_FOLDER, '*.json')
    
    jsons = {}
    for f in glob.glob(input_glob):
        f_name = os.path.basename(f)
        host = f_name.split('.')[0]
        
        with open(f) as json_file:
            jsons[int(host)] = json.load(json_file)
    
    pages = OrderedDict()
    for host in sorted(jsons):
        for page in xrange(len(jsons[host]['pages'])):
            page_vals = OrderedDict()
            pages[jsons[host]['pages'][page]['page_id']] = page_vals

            page_vals['host_id'] = host 
            
            for page_attr in sorted(jsons[host]['pages'][page]):
                if 'page_id' == page_attr:
                    continue
                if 'series_48h' == page_attr:
                    continue

                if 'series' in page_attr:
                    series_dict = jsons[host]['pages'][page][page_attr]
                    series_names = sorted(series_dict)
            
                    for series_name in series_names:
                        n_ticks = len(series_dict[series_name])
                        for t in xrange(n_ticks):
                            page_vals['series_%s_%d' % (series_name, t)] = \
                                    series_dict[series_name][t]
                else:
                    page_vals[page_attr] = \
                            jsons[host]['pages'][page][page_attr]
    
    return pd.DataFrame.from_dict(pages, orient='index')

if __name__ == '__main__':
    df = to_pandas()
    df.to_csv(os.path.join(DATAFRAME_FOLDER, TRAIN_CSV_FPATH))
    
    df = to_pandas(True)
    df.to_csv(os.path.join(DATAFRAME_FOLDER, TEST_CSV_FPATH))
    
    for series_name in ['visits', 'facebook', 'twitter']:
        df = to_48h_tseries_mat(series_name=series_name)
        df.to_csv(os.path.join(DATAFRAME_FOLDER, \
                _48H_SERIES_NAMES % series_name))
