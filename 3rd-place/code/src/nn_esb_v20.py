
# -*- coding: utf-8 -*-

import datetime
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
from sklearn import cross_validation as cv
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import StandardScaler
from sklearn.externals import joblib
from visualise_data import SequenceVisualisation

import json
import logging
import os

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import Adam
from keras.utils import np_utils

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

def keras_model(sz=1000):
    input_dim = sz
    adam = Adam(lr=5e-6, beta_1=0.9, beta_2=0.999, epsilon=1e-8)
    model = Sequential()
    model.add(Dense(1024, input_shape=(input_dim,)))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1024))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(20))
    model.add(Activation('softmax'))
    
    model.compile(loss='categorical_crossentropy',
                  optimizer=adam,
                  metrics=['accuracy'])
    return model

def pred_binary(xte,clf):
    prediction  = clf.predict_proba(xte) 
    return prediction
     
def brier_score(given, predicted):
    global class_weights
    return np.power(given - predicted, 2.0).dot(class_weights).mean()


def predict(model, test_x, filename):
    
    probs = model.predict_proba(test_x)

    df = pd.read_csv("../sub/xgb_v5.tst.csv")
    df[df.columns[3:]] = probs
    df.to_csv(filename, index=False)        
    
def nn_model(train_x, train_y, nb_epoch=100, batch_size=128):
   
    logging.info("Modelling with "+ str(train_x.shape[1]) + " features ...")

    # Cross Validation
    logging.info("Cross validation... ")
    kfold = 5
    y_binary = np.argmax(train_y, axis=1)
    
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
    
         classifier = keras_model(sz=X_train.shape[1])
         dummy_y = np_utils.to_categorical(y_train)
         classifier.fit(X_train,dummy_y, nb_epoch=nb_epoch, batch_size=batch_size, 
                        validation_data=[X_test, y_test],
                        verbose=1)

         y_predict=pred_binary(X_test,classifier)
  
         val_preds[test_indices,:] = y_predict
        
         # Compute confusion matrix
         score = brier_score(y_test, y_predict)
         cv_score = cv_score + score
         logging.info("fold #%d: %f" % (i,score))
         models.append(classifier)

    logging.info("%d fold: %f" % (kfold, cv_score/kfold))
    np.savetxt("../models/nn_esb_v20.val.txt", val_preds)
    
    #fit all data
    classifier = keras_model(sz=train_x.shape[1])
    dummy_y = np_utils.to_categorical(y_binary)
    classifier.fit(train_x,dummy_y, nb_epoch=nb_epoch, batch_size=batch_size, verbose=1)
    return classifier, models, 0.167646

def main():
    logging.info("Loading data - " + str(datetime.datetime.now()))
    train_x, train_y, test_x = joblib.load("../input/esb20.dmp")
    
    logging.info("Pre processing data - " + str(datetime.datetime.now()))
    logging.info("Building model - " + str(datetime.datetime.now()))
    clf, models, score = nn_model(train_x, train_y)
    logging.info("Predicting test data - " + str(datetime.datetime.now()))
    predict(clf, test_x, "../sub/nn_esb_v20.tst.csv")

if __name__ == "__main__":
    main()
