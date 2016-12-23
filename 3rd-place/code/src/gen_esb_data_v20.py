
# -*- coding: utf-8 -*-

import datetime
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import StandardScaler
from sklearn.externals import joblib
from visualise_data import SequenceVisualisation

import json
import logging
logging.basicConfig(format='%(asctime)s   %(levelname)s   %(message)s',
                        level=logging.DEBUG)
seed = 2016
plot = True
imputer = Imputer()
scaler = StandardScaler()

activity_names = json.load(open('../input/public_data/annotations.json', 'r'))
class_weights = np.asarray(json.load(open('../input/public_data/class_weights.json', 'r')))
plotter = SequenceVisualisation('../input/public_data', '../input/public_data/train/00001')
annotation_names = plotter.targets.columns.difference(['start', 'end'])

def brier_score(given, predicted):
    global class_weights
    return np.power(given - predicted, 2.0).dot(class_weights).mean()

def load_sequence(file_id):
    filename = str(file_id).zfill(5)
    
    target = np.asarray(pd.read_csv('../input/public_data/train/{}/targets.csv'.format(filename)))[:, 2:]

    return target

def load_test_data():
    test_x = []
    file_list = ["xgb_v5.tst.csv", 
                 "xgb_v8.tst.csv", 
                 "knn_v5.tst.csv",
                 "xgb_v10.tst.csv",
                 "xgb_v12.tst.csv",
                 "xgb_v14.tst.csv",
                 "xgb_v15.tst.csv",
                 "xgb_v16.tst.csv",
                 "xgb_v18.tst.csv",
                 "xgb_v19.tst.csv",
                 "xgb_v21.tst.csv",
                 "xgb_v20.tst.csv",
                 "xgb_v22.tst.csv",
                 ]
    for f in file_list:
        df = pd.read_csv("../sub/%s" % f)
        df = df[df.columns[3:]]
        test_x.append(df.values)

    test_x = np.hstack(test_x)
    
    test_x1 = np.zeros(test_x.shape)
    test_x1[0,:] = -1
    test_x1[1:,:] = test_x[:-1,:]
    
    test_x2 = np.zeros(test_x.shape)
    test_x2[0,:] = -1
    test_x2[1:,:] = test_x1[:-1,:]
    
    test_x3 = np.zeros(test_x.shape)
    test_x3[-1,:] = -1
    test_x3[:-1,:] = test_x[1:,:]
    
    test_x4 = np.zeros(test_x.shape)
    test_x4[-1,:] = -1
    test_x4[:-1,:] = test_x3[1:,:]
    
    test_x5 = np.zeros(test_x.shape)
    test_x5[0,:] = -1
    test_x5[1:,:] = test_x2[:-1,:]
    
    test_x6 = np.zeros(test_x.shape)
    test_x6[-1,:] = -1
    test_x6[:-1,:] = test_x4[1:,:]
    
    test_x = np.hstack((test_x, test_x1, test_x2, test_x3, test_x4, test_x5, test_x6))
    
    return test_x
    
def load_sequences(file_ids):
    train_x = []
    file_list = ["xgb_v5.val.txt", 
                 "xgb_v8.val.txt", 
                 "knn_v5.val.txt",
                 "xgb_v10.val.txt", 
                 "xgb_v12.val.txt",
                 "xgb_v14.val.txt",
                 "xgb_v15.val.txt",
                 "xgb_v16.val.txt",
                 "xgb_v18.val.txt",
                 "xgb_v19.val.txt",
                 "xgb_v21.val.txt",
                 "xgb_v20.val.txt",
                 "xgb_v22.val.txt",
                 ]
    for f in file_list:
        train_x.append(np.loadtxt("../models/%s" % f))
        
    train_x = np.hstack(train_x)
    
    y_es = []
    for file_id in file_ids:
        target = load_sequence(file_id)
        y_es.append(target)

    return train_x, np.row_stack(y_es)

def load_train_data():
    # Load the training and testing data
    train_x, train_y = load_sequences([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

    # We will want to impute the missing data
    global imputer
    global scaler

    """
    Note, not all data is annotated, so we select only the annotated rows
    """
    train_y_has_annotation = np.isfinite(train_y.sum(1))
    train_y = train_y[train_y_has_annotation]

    train_x1 = np.zeros(train_x.shape)
    train_x1[0,:] = -1
    train_x1[1:,:] = train_x[:-1,:]
    
    train_x2 = np.zeros(train_x.shape)
    train_x2[0,:] = -1
    train_x2[1:,:] = train_x1[:-1,:]
    
    train_x3 = np.zeros(train_x.shape)
    train_x3[-1,:] = -1
    train_x3[:-1,:] = train_x[1:,:]
    
    train_x4 = np.zeros(train_x.shape)
    train_x4[-1,:] = -1
    train_x4[:-1,:] = train_x3[1:,:]
    
    train_x5 = np.zeros(train_x.shape)
    train_x5[0,:] = -1
    train_x5[1:,:] = train_x2[:-1,:]
    
    train_x6 = np.zeros(train_x.shape)
    train_x6[-1,:] = -1
    train_x6[:-1,:] = train_x4[1:,:]
    
    train_x = np.hstack((train_x, train_x1, train_x2, train_x3, train_x4, train_x5, train_x6))
    
    logging.info ("Training data shapes:")
    logging.info ("train_x.shape: {}".format(train_x.shape))
    logging.info ("train_y.shape: {}".format(train_y.shape))

    return (train_x,train_y)
    
def main():
    logging.info("Loading data - " + str(datetime.datetime.now()))
    train_x, train_y = load_train_data()
    test_x = load_test_data()
    joblib.dump([train_x, train_y, test_x], "../input/esb20.dmp")
    

if __name__ == "__main__":
    main()
