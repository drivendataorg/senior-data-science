
# -*- coding: utf-8 -*-

import datetime
import warnings

import numpy as np
import pandas as pd
import xgboost as xgb

warnings.filterwarnings("ignore")
from sklearn import cross_validation as cv
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import StandardScaler

from visualise_data import SequenceVisualisation

import time
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
                 "rf_v13.tst.csv",
                 "xgb_v20.tst.csv",
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
    
    test_x = np.hstack((test_x, test_x1, test_x2, test_x3, test_x4))
    
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
                 "rf_v13.val.txt",
                 "xgb_v20.val.txt",
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
    
    train_x = np.hstack((train_x, train_x1, train_x2, train_x3, train_x4))
    
    logging.info ("Training data shapes:")
    logging.info ("train_x.shape: {}".format(train_x.shape))
    logging.info ("train_y.shape: {}".format(train_y.shape))

    return (train_x,train_y)
    
def predict(model, test_x, filename, isXgb=False):
    if isinstance(model, list):
        probs = 0
        for m in model:
            if isXgb:
                xg_test = xgb.DMatrix(test_x)
                probs += m.predict(xg_test).reshape(test_x.shape[0], 20)
            else:
                probs += m.predict_proba(test_x)
                
        probs /= len(model)                        
    else:
        if isXgb:
            xg_test = xgb.DMatrix(test_x)
            probs = model.predict(xg_test).reshape(test_x.shape[0], 20)
        else:
            probs = model.predict_proba(test_x)

    df = pd.read_csv("../sub/xgb_v5.tst.csv")
    df[df.columns[3:]] = probs
    df.to_csv(filename, index=False)        

def xgb_model(train_x, train_y):
    depth = 6
    eta = 0.01
    ntrees = 600

    params = {"objective": "multi:softprob",
               "num_class": 20,
               "booster": "gbtree",
               "eta": eta,
               "max_depth": depth,
               "min_child_weight": 10,
               "subsample": 0.7,
               "colsample_bytree": 0.7,
               "eval_metric": "mlogloss",
               "silent": 1}

    logging.info("Modelling with ntrees: " + str(ntrees))
    logging.info("Modelling with "+ str(train_x.shape[1]) + " features ...")


    # Cross Validation
    logging.info("Cross validation... ")
    kfold = 5
    y_binary = np.argmax(train_y, axis=1)

    weights = np.zeros(train_x.shape[0])
    '''
    for c in np.unique(y_binary):
        weights[y_binary==c] = class_weights[c]
    '''
    weights = train_y.max(axis=1)
    
    skf = cv.StratifiedKFold(y_binary, kfold, random_state=2016)
    skfind = [None]*len(skf) # indices
    cnt=0
    cv_score = 0.0
    for train_index in skf:
      skfind[cnt] = train_index
      cnt = cnt + 1

    models = []
    val_preds = np.zeros((train_x.shape[0], 20))
    for i in range(kfold):
         train_indices = skfind[i][0]
         test_indices = skfind[i][1]
    
         X_train = train_x[train_indices]
         y_train = train_y[train_indices]
         y_train = np.argmax(y_train, axis=1)
         X_test = train_x[test_indices]
         y_test = train_y[test_indices]
    
         weight_train = weights[train_indices]
         weight_test = weights[test_indices]
         
         tic = time.time()
         xg_train = xgb.DMatrix(X_train, label=y_train, weight=weight_train)
         xg_val = xgb.DMatrix(X_test, label=np.argmax(y_test, axis=1), weight=weight_test)
         
         xg_test = xgb.DMatrix(X_test)
    
         watchlist = [(xg_train, 'train'), (xg_val, 'val')]
         classifier = xgb.train(params, xg_train, ntrees, watchlist)
         toc = time.time()
         print "training time= ", (toc-tic)
    
         # Testing
         y_predict = []
         tic = time.time()
         y_predict = classifier.predict(xg_test).reshape(y_test.shape[0], 20 )
         toc = time.time()
         print "testing time = ", (toc-tic)
    
         val_preds[test_indices,:] = y_predict
        
         # Compute confusion matrix
         score = brier_score(y_test, y_predict)
         cv_score = cv_score + score
         logging.info("fold #%d: %f" % (i,score))
         models.append(classifier)

    logging.info("%d fold: %f" % (kfold, cv_score/kfold))
    np.savetxt("../models/xgb_esb_v3.val.txt", val_preds)
    
    #fit all data
    xg_train = xgb.DMatrix(train_x, label=y_binary)
    classifier = xgb.train(params, xg_train, ntrees)
    return classifier, models, cv_score/kfold

def main():
    logging.info("Loading data - " + str(datetime.datetime.now()))
    train_x, train_y = load_train_data()
    test_x = load_test_data()
    
    logging.info("Pre processing data - " + str(datetime.datetime.now()))
    logging.info("Building model - " + str(datetime.datetime.now()))
    clf, models, score = xgb_model(train_x,train_y)
    logging.info("Predicting test data - " + str(datetime.datetime.now()))
    predict(clf, test_x, "../sub/xgb_esb_v3.tst.csv", True)

if __name__ == "__main__":
    main()
