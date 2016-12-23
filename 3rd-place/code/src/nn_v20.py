
# -*- coding: utf-8 -*-

import datetime
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
from sklearn import cross_validation as cv
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import StandardScaler

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
    model.add(Dense(512, input_shape=(input_dim,)))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(512))
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

def load_sequence(file_id):
    filename = str(file_id).zfill(5)
    df1 = pd.read_csv('../input/public_data/train/{}/columns_v5.csv'.format(filename))
    df2 = pd.read_csv('../input/public_data/train/{}/columns_v7.csv'.format(filename))
    df3 = pd.read_csv('../input/public_data/train/{}/columns_v8.csv'.format(filename))
    
    data = np.hstack((df1.values, df2.values, df3.values))
    target = np.asarray(pd.read_csv('../input/public_data/train/{}/targets.csv'.format(filename)))[:, 2:]

    return data, target

def load_sequences(file_ids):
    x_es = []
    y_es = []

    for file_id in file_ids:
        data, target = load_sequence(file_id)

        x_es.append(data)
        y_es.append(target)

    return np.row_stack(x_es), np.row_stack(y_es)

def load_data():
    # Load the training and testing data
    train_x, train_y = load_sequences([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

    # We will want to impute the missing data
    global imputer
    global scaler
    imputer.fit(train_x)
    train_x = imputer.transform(train_x)

    """
    Note, not all data is annotated, so we select only the annotated rows
    """
    train_y_has_annotation = np.isfinite(train_y.sum(1))
    train_x = train_x[train_y_has_annotation]
    train_y = train_y[train_y_has_annotation]

    """
    logging.info simple statistics regarding the number of instances
    """
    logging.info ("Training data shapes:")
    logging.info ("train_x.shape: {}".format(train_x.shape))
    logging.info ("train_y.shape: {}".format(train_y.shape))


    scaler = scaler.fit(train_x)
    train_x = scaler.transform(train_x)

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
    
    return (train_x,train_y)

def process_data(train_x):
    #shuffle
    #train = np.random.shuffle(train)
    '''
    x_train_normalized = normalize(train_x)

    pca = PCA(n_components=150)
    train_x_projected_pca = pca.fit_transform(x_train_normalized)

    rmb = BernoulliRBM(n_components=50)
    train_x_projected_rrm = rmb.fit_transform(x_train_normalized)

    km = KMeans(n_clusters=20)
    train_x_projected_km = km.fit_transform(x_train_normalized)
    train_x_projected = np.hstack((train_x_projected_pca, train_x_projected_rrm, train_x_projected_km))
    '''
    return (train_x)

def load_test_data():
    test_x = []
    for te_ind_str in sorted(os.listdir(os.path.join('../input/public_data', 'test'))):

        features_v5 = pd.read_csv(os.path.join('../input/public_data', 'test', te_ind_str, 'columns_v5.csv')).values
        features_v7 = pd.read_csv(os.path.join('../input/public_data', 'test', te_ind_str, 'columns_v7.csv')).values
        features_v8 = pd.read_csv(os.path.join('../input/public_data', 'test', te_ind_str, 'columns_v8.csv')).values

        features = np.hstack((features_v5, features_v7, features_v8))

        features = imputer.transform(features)
        features = scaler.transform(features)

        features = process_data(features)
        
        features1 = np.zeros(features.shape)
        features1[0,:] = -1
        features1[1:,:] = features[:-1,:]

        features2 = np.zeros(features.shape)
        features2[0,:] = -1
        features2[1:,:] = features1[:-1,:]
        
        features3 = np.zeros(features.shape)
        features3[-1,:] = -1
        features3[:-1:,:] = features[1:,:]
        
        features4 = np.zeros(features.shape)
        features4[-1,:] = -1
        features4[:-1:,:] = features3[1:,:]
        
        features = np.hstack((features, features1, features2, features3, features4))
        test_x.append(features)
    
    return np.row_stack(test_x)

def load_train_data():
    # Load the training and testing data
    train_x, train_y = load_data()

    return (train_x,train_y)
    

def predict(model, filename):
    num_lines = 0
    se_cols = ['start', 'end']

    with open(filename, 'w') as fil:
        fil.write(','.join(['record_id'] + se_cols + annotation_names.tolist()))
        fil.write('\n')

        for te_ind_str in sorted(os.listdir(os.path.join('../input/public_data', 'test'))):
            te_ind = int(te_ind_str)

            meta = json.load(open(os.path.join('../input/public_data', 'test', te_ind_str, 'meta.json')))
            features_v5 = pd.read_csv(os.path.join('../input/public_data', 'test', te_ind_str, 'columns_v5.csv')).values
            features_v7 = pd.read_csv(os.path.join('../input/public_data', 'test', te_ind_str, 'columns_v7.csv')).values
            features_v8 = pd.read_csv(os.path.join('../input/public_data', 'test', te_ind_str, 'columns_v8.csv')).values

            features = np.hstack((features_v5, features_v7, features_v8))
    
            features = imputer.transform(features)
            features = scaler.transform(features)

            features = process_data(features)
            
            features1 = np.zeros(features.shape)
            features1[0,:] = -1
            features1[1:,:] = features[:-1,:]
    
            features2 = np.zeros(features.shape)
            features2[0,:] = -1
            features2[1:,:] = features1[:-1,:]
            
            features3 = np.zeros(features.shape)
            features3[-1,:] = -1
            features3[:-1:,:] = features[1:,:]
            
            features4 = np.zeros(features.shape)
            features4[-1,:] = -1
            features4[:-1:,:] = features3[1:,:]
            
            features = np.hstack((features, features1, features2, features3, features4))

            probs = model.predict_proba(features)

            starts = range(meta['end'])
            ends = range(1, meta['end'] + 1)

            for start, end, prob in zip(starts, ends, probs):
                row = [te_ind, start, end] + prob.tolist()

                fil.write(','.join(map(str, row)))
                fil.write('\n')

                num_lines += 1

    logging.info("{} lines written.".format(num_lines))
    
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
         classifier.fit(X_train,dummy_y, nb_epoch=nb_epoch, batch_size=batch_size, verbose=1)

         y_predict=pred_binary(X_test,classifier)
  
         val_preds[test_indices,:] = y_predict
        
         # Compute confusion matrix
         score = brier_score(y_test, y_predict)
         cv_score = cv_score + score
         logging.info("fold #%d: %f" % (i,score))
         models.append(classifier)

    logging.info("%d fold: %f" % (kfold, cv_score/kfold))
    np.savetxt("../models/%f_nn_v20.val.txt" % 0.194267, val_preds)
    
    #fit all data
    classifier = keras_model(sz=train_x.shape[1])
    dummy_y = np_utils.to_categorical(y_binary)
    classifier.fit(train_x,dummy_y, nb_epoch=nb_epoch, batch_size=batch_size, verbose=1)
    return classifier, models, 0.194267

def main():
    logging.info("Loading data - " + str(datetime.datetime.now()))
    train_x, train_y = load_train_data()
    
    logging.info("Pre processing data - " + str(datetime.datetime.now()))
    logging.info("Building model - " + str(datetime.datetime.now()))
    clf, models, score = nn_model(train_x,train_y)
    logging.info("Predicting test data - " + str(datetime.datetime.now()))
    predict(clf, "../sub/%f_nn_v20.tst.csv"%score)

if __name__ == "__main__":
    main()
