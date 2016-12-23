# -*- coding: utf-8 -*-
"""
Created on Wed Jul  6 08:33:41 2016

"""

# For number crunching
import numpy as np
import pandas as pd

# For prediction
from sklearn.ensemble import ExtraTreesRegressor


# Misc
import time

def brier_score(given, predicted, weight_vector):
    return np.power(given - predicted, 2.0).dot(weight_vector).mean()


def L1_train(dataset, preprocess, predict_model, class_weights=None, verbose=1):

    # Start
    all_train_x = dataset['train_x']
    all_train_y = dataset['train_y']
    train_seq = dataset['train_seq']
    start_time = time.time()
    if class_weights is None:
        class_weights = np.repeat(1, all_train_y.shape[1])

    # Main
    scores = []
    for i in range(10):

        if verbose > 1:
            print ("")
            print ("Training Fold Number {}".format(i+1))

        # Create sequences
        sequence_train = [1,2,3,4,5,6,7,8,9,10]
        sequence_test = [sequence_train.pop(i)]

        # Load the training and testing data
        if verbose > 1:
            tmp_start_time = time.time()
            print ("... loading data"),
        train_x = all_train_x[np.in1d(np.array(train_seq), np.array(sequence_train))]
        train_y = all_train_y[np.in1d(np.array(train_seq), np.array(sequence_train))]
        test_x = all_train_x[np.in1d(np.array(train_seq), np.array(sequence_test))]
        test_y = all_train_y[np.in1d(np.array(train_seq), np.array(sequence_test))]
        if verbose > 1:
            print (": {:.1f} s".format((time.time() - tmp_start_time)))

        # preprocess
        if verbose > 1:
            tmp_start_time = time.time()
            print ("... preprocessing data"),
        train_x, train_y, test_x = preprocess(train_x, train_y, test_x)
        if verbose > 1:
            print (": {:.1f} s".format((time.time() - tmp_start_time)))

        # Predict on the test instances
        if verbose > 1:
            tmp_start_time = time.time()
            print ("... training & predicting"),
        test_predicted = predict_model(train_x, train_y, test_x, test_y)
        if verbose > 1:
            print (": {:.2f} min".format((time.time() - tmp_start_time)/60))
        score = brier_score(test_y, test_predicted, class_weights)
        if verbose > 0:
            print ("    Fold {} score = {:.5f}".format(i+1, score))
        scores.append(score)

        if i == 0:
            valid_predicted = test_predicted
            valid_y = test_y
        else:
            valid_predicted = np.concatenate((valid_predicted, test_predicted), axis=0)
            valid_y = np.concatenate((valid_y, test_y), axis=0)

    scores = np.asarray(scores)
    if verbose > 0:
        print ("")
        print ("Mean score = {:.5f}".format(scores.mean()))
        print ("Total score = {:.5f}".format(brier_score(valid_y, valid_predicted, class_weights)))
        print ("Total execution time: {:.1f} min".format((time.time() - start_time)/60))

    return valid_predicted, scores


def L1_train_aug(dataset, preprocess, predict_model, class_weights=None, verbose=1):

    # Start
    all_train_x = dataset['train_x']
    all_train_y = dataset['train_y']
    train_seq = dataset['train_seq']

    # Augmented train data
    aug_train_x = dataset['aug_train_x']
    aug_train_y = dataset['aug_train_y']
    aug_train_seq = dataset['aug_train_seq']

    start_time = time.time()
    if class_weights is None:
        class_weights = np.repeat(1, all_train_y.shape[1])

    # Main
    scores = []
    for i in range(10):

        if verbose > 1:
            print ("")
            print ("Training Augmented data for Fold Number {}".format(i+1))

        # Create sequences
        sequence_train = [1,2,3,4,5,6,7,8,9,10]
        sequence_test = [sequence_train.pop(i)]

        # Load the training and testing data
        if verbose > 1:
            tmp_start_time = time.time()
            print ("... loading data"),
        train_x = aug_train_x[np.in1d(np.array(aug_train_seq), np.array(sequence_train))]
        train_y = aug_train_y[np.in1d(np.array(aug_train_seq), np.array(sequence_train))]
        test_x = all_train_x[np.in1d(np.array(train_seq), np.array(sequence_test))]
        test_y = all_train_y[np.in1d(np.array(train_seq), np.array(sequence_test))]
        if verbose > 1:
            print (": {:.1f} s".format((time.time() - tmp_start_time)))

        # preprocess
        if verbose > 1:
            tmp_start_time = time.time()
            print ("... preprocessing data"),
        train_x, train_y, test_x = preprocess(train_x, train_y, test_x)
        if verbose > 1:
            print (": {:.1f} s".format((time.time() - tmp_start_time)))

        # Predict on the test instances
        if verbose > 1:
            tmp_start_time = time.time()
            print ("... training & predicting"),
        test_predicted = predict_model(train_x, train_y, test_x, test_y)
        if verbose > 1:
            print (": {:.2f} min".format((time.time() - tmp_start_time)/60))
        score = brier_score(test_y, test_predicted, class_weights)
        if verbose > 0:
            print ("    Fold {} score = {:.5f}".format(i+1, score))
        scores.append(score)

        if i == 0:
            valid_predicted = test_predicted
            valid_y = test_y
        else:
            valid_predicted = np.concatenate((valid_predicted, test_predicted), axis=0)
            valid_y = np.concatenate((valid_y, test_y), axis=0)

    scores = np.asarray(scores)
    if verbose > 0:
        print ("")
        print ("Mean score = {:.5f}".format(scores.mean()))
        print ("Total score = {:.5f}".format(brier_score(valid_y, valid_predicted, class_weights)))
        print ("Total execution time: {:.1f} min".format((time.time() - start_time)/60))

    return valid_predicted, scores


def L1_trial_aug(dataset, fold, preprocess, predict_model, class_weights=None, verbose=1):

    # Start
    start_time = time.time()
    all_train_x = dataset['train_x']
    all_train_y = dataset['train_y']
    train_seq = dataset['train_seq']

    # Augmented train data
    aug_train_x = dataset['aug_train_x']
    aug_train_y = dataset['aug_train_y']
    aug_train_seq = dataset['aug_train_seq']


    if class_weights is None:
        class_weights = np.repeat(1, all_train_y.shape[1])
    i = fold - 1

    # Main
    if verbose > 1:
        print ("")
        print ("Training Fold Number {}".format(i+1))

    # Create sequences
    sequence_train = [1,2,3,4,5,6,7,8,9,10]
    sequence_test = [sequence_train.pop(i)]

    # Load the training and testing data
    if verbose > 1:
        tmp_start_time = time.time()
        print ("... loading data"),
    train_x = aug_train_x[np.in1d(np.array(aug_train_seq), np.array(sequence_train))]
    train_y = aug_train_y[np.in1d(np.array(aug_train_seq), np.array(sequence_train))]
    test_x = all_train_x[np.in1d(np.array(train_seq), np.array(sequence_test))]
    test_y = all_train_y[np.in1d(np.array(train_seq), np.array(sequence_test))]
    if verbose > 1:
        print (": {:.1f} s".format((time.time() - tmp_start_time)))

    # preprocess
    if verbose > 1:
        tmp_start_time = time.time()
        print ("... preprocessing data"),
    train_x, train_y, test_x = preprocess(train_x, train_y, test_x)
    if verbose > 1:
        print (": {:.1f} s".format((time.time() - tmp_start_time)))

    # Predict on the test instances
    if verbose > 1:
        tmp_start_time = time.time()
        print ("... training & predicting"),
    test_predicted = predict_model(train_x, train_y, test_x, test_y)
    if verbose > 1:
        print (": {:.2f} min".format((time.time() - tmp_start_time)/60))
    score = brier_score(test_y, test_predicted, class_weights)
    if verbose > 0:
        print ("    Fold {} score = {:.5f}".format(i+1, score))
        print ("    Execution time: {:.1f} min".format((time.time() - start_time)/60))


def L1_trial(dataset, fold, preprocess, predict_model, class_weights=None, verbose=1):

    # Start
    start_time = time.time()
    all_train_x = dataset['train_x']
    all_train_y = dataset['train_y']
    train_seq = dataset['train_seq']
    if class_weights is None:
        class_weights = np.repeat(1, all_train_y.shape[1])
    i = fold - 1

    # Main
    if verbose > 1:
        print ("")
        print ("Training Augmented data for Fold Number {}".format(i+1))

    # Create sequences
    sequence_train = [1,2,3,4,5,6,7,8,9,10]
    sequence_test = [sequence_train.pop(i)]

    # Load the training and testing data
    if verbose > 1:
        tmp_start_time = time.time()
        print ("... loading data"),
    train_x = all_train_x[np.in1d(np.array(train_seq), np.array(sequence_train))]
    train_y = all_train_y[np.in1d(np.array(train_seq), np.array(sequence_train))]
    test_x = all_train_x[np.in1d(np.array(train_seq), np.array(sequence_test))]
    test_y = all_train_y[np.in1d(np.array(train_seq), np.array(sequence_test))]
    if verbose > 1:
        print (": {:.1f} s".format((time.time() - tmp_start_time)))

    # preprocess
    if verbose > 1:
        tmp_start_time = time.time()
        print ("... preprocessing data"),
    train_x, train_y, test_x = preprocess(train_x, train_y, test_x)
    if verbose > 1:
        print (": {:.1f} s".format((time.time() - tmp_start_time)))

    # Predict on the test instances
    if verbose > 1:
        tmp_start_time = time.time()
        print ("... training & predicting"),
    test_predicted = predict_model(train_x, train_y, test_x, test_y)
    if verbose > 1:
        print (": {:.2f} min".format((time.time() - tmp_start_time)/60))
    score = brier_score(test_y, test_predicted, class_weights)
    if verbose > 0:
        print ("    Fold {} score = {:.5f}".format(i+1, score))
        print ("    Execution time: {:.1f} min".format((time.time() - start_time)/60))


def L1_test(dataset, preprocess, predict_model, verbose=1):

    # start
    all_train_x = dataset['train_x']
    all_train_y = dataset['train_y']
    all_test_x = dataset['test_x']

    # preprocess
    all_train_x, all_train_y, all_test_x = preprocess(all_train_x, all_train_y, all_test_x)

    # Predict on the test instances
    test_predicted = predict_model(all_train_x, all_train_y, all_test_x)

    return test_predicted


def L1_test_aug(dataset, preprocess, predict_model, verbose=1):

    # start
    all_train_x = dataset['aug_train_x']
    all_train_y = dataset['aug_train_y']
    all_test_x = dataset['test_x']

    # preprocess
    all_train_x, all_train_y, all_test_x = preprocess(all_train_x, all_train_y, all_test_x)

    # Predict on the test instances
    test_predicted = predict_model(all_train_x, all_train_y, all_test_x)

    return test_predicted

class PMC_ExtraTreesRegressor(): # Probabilistic MultiClassification

    def __init__(self, nb_classes, bags=1, params={}):

        # params
        self.params = params

        #common
        self.nb_classes = nb_classes
        self.bags = bags
        self.bags_models = tuple()
        self.train_y = None

        for bag in range(self.bags):
            models = tuple()
            for k in range(self.nb_classes):
                model = ExtraTreesRegressor()
                model.set_params(**self.params)
                model.set_params(random_state = (self.params['random_state'] + bag))
                models = models + (model,)
            self.bags_models = self.bags_models + (models, )

    def fit(self, train_x, train_y):
        self.train_y = train_y
        for bag in range(self.bags):
            for k in range(self.nb_classes):
                self.bags_models[bag][k].fit(train_x, train_y[:,k])

    def predict_proba(self, test_x):
        for bag in range(self.bags):
            for k in range(self.nb_classes):
                predict = self.bags_models[bag][k].predict(test_x)
                predict = predict.reshape((predict.shape[0], 1))

                # concatenate predictions
                if k==0:
                    test_predicted = predict
                else:
                    test_predicted = np.concatenate((test_predicted, predict), axis=1)

            # limitate predictions to [0,1]
            test_predicted = np.apply_along_axis(lambda x:map( lambda y:max(y,0),x), 0, test_predicted)
            test_predicted = np.apply_along_axis(lambda x:map( lambda y:min(y,1),x), 0, test_predicted)

            # normalize class probabilities
            test_predicted = np.apply_along_axis(lambda x:x/sum(x), 1, test_predicted)

            if bag == 0:
                total_predicted = test_predicted
            else:
                total_predicted = total_predicted + test_predicted

        total_predicted = total_predicted / self.bags

        return test_predicted


class PMC_MultiTaskExtraTreesRegressor(): # Probabilistic MultiClassification

    def __init__(self, nb_classes, bags=1, params={}):

        # params
        self.params = params

        #common
        self.nb_classes = nb_classes
        self.bags = bags
        self.bags_models = tuple()
        self.train_y = None

        for bag in range(self.bags):
            model = ExtraTreesRegressor()
            model.set_params(**self.params)
            model.set_params(random_state = (self.params['random_state'] + bag))
            self.bags_models = self.bags_models + (model, )

    def fit(self, train_x, train_y, sample_weight = None):
        self.train_y = train_y
        for bag in range(self.bags):
            self.bags_models[bag].fit(train_x, train_y, sample_weight)

    def predict_proba(self, test_x):
        for bag in range(self.bags):
            test_predicted = self.bags_models[bag].predict(test_x)

            # limitate predictions to [0,1]
            test_predicted = np.apply_along_axis(lambda x:map( lambda y:max(y,0),x), 0, test_predicted)
            test_predicted = np.apply_along_axis(lambda x:map( lambda y:min(y,1),x), 0, test_predicted)

            # normalize class probabilities
            test_predicted = np.apply_along_axis(lambda x:x/sum(x), 1, test_predicted)

            if bag == 0:
                total_predicted = test_predicted
            else:
                total_predicted = total_predicted + test_predicted

        total_predicted = total_predicted / self.bags

        return test_predicted


class PMC_NeuralNetwork(object):
    def __init__(self, dims, nb_classes, bags=1):
        #common
        self.nb_classes = nb_classes
        self.bags = bags
        self.dims = dims

    def fit(self, train_x, train_y, test_x = None, test_y = None, batch_size=128, nb_epoch=1):
        for bag in range(self.bags):
            if test_y is not None:
                self.bags_models[bag].fit(train_x, train_y, nb_epoch=nb_epoch, batch_size=batch_size,
                                          shuffle=True, validation_data=(test_x, test_y), verbose=0)
            else:
                self.bags_models[bag].fit(train_x, train_y, nb_epoch=nb_epoch, batch_size=batch_size,
                                          shuffle=True, verbose=0)

    def predict_proba(self, test_x):
        for bag in range(self.bags):
            test_predicted = self.bags_models[bag].predict_proba(test_x, verbose=0)
            if bag == 0:
                total_predicted = test_predicted
            else:
                total_predicted = total_predicted + test_predicted
        total_predicted = total_predicted / self.bags
        return(total_predicted)


def loss_brier_score(y_true, y_pred):
    from keras import backend as K
    return K.mean(K.sum(K.pow(y_pred - y_true, 2), axis=-1), axis=0)

def keras_brier(y_true, y_pred):
    from keras import backend as K
    weight = [ 1.35298456,  1.38684574,  1.59587388,  1.35318714,  0.34778367,
        0.66108171,  1.04723629,  0.39886522,  0.20758632,  1.50578335,
        0.11018137,  1.07803284,  1.36560417,  1.17024114,  1.19336374,
        1.18037045,  1.34414875,  1.11683831,  1.0808391 ,  0.50315225]

    return K.mean(K.dot(K.square(y_pred - y_true), weight), axis=-1)

class PMC_NeuralNetwork_T1(PMC_NeuralNetwork):
    def __init__(self, dims, nb_classes, params={}, bags=1):
        # libraries
        from keras.models import Sequential
        from keras.layers.core import Dense, Dropout, Activation
        from keras.layers.normalization import BatchNormalization
        from keras.layers.advanced_activations import PReLU
        from keras.optimizers import Adagrad,SGD,Adadelta,Adam

        super(PMC_NeuralNetwork_T1, self).__init__(dims, nb_classes, bags)
        self.layers = params.get('layers', [[100, 0.50], [100, 0.50]])
        self.loss = params.get('loss', 'categorical_crossentropy')
        if self.loss == 'brier':
            self.loss = keras_brier
        self.bags_models = tuple()
        #add models
        for bag in range(self.bags):
            model = Sequential()
            for i_layer in self.layers:
                model.add(Dense(i_layer[0], input_shape=(dims,)))
                model.add(PReLU())
                model.add(BatchNormalization())
                model.add(Dropout(i_layer[1]))
            model.add(Dense(nb_classes))
            model.add(Activation('softmax'))
            opt=Adam()
            model.compile(loss=self.loss, optimizer=opt)
            self.bags_models = self.bags_models + (model, )


class PMC_LinearRegression(): # Probabilistic MultiClassification
    def __init__(self, nb_classes, bags=1, param={}):

        # libraries
        from sklearn.linear_model import LinearRegression

        # params
        self.n_jobs = param.get('n_jobs',-1)

        #common
        self.nb_classes = nb_classes
        self.bags = bags
        self.bags_models = tuple()
        self.train_y = None
        for bag in range(self.bags):
            models = tuple()
            for k in range(self.nb_classes):
                model = LinearRegression(n_jobs = self.n_jobs)
                models = models + (model,)
            self.bags_models = self.bags_models + (models, )

    def fit(self, train_x, train_y):
        self.train_y = train_y
        for bag in range(self.bags):
            for k in range(self.nb_classes):
                self.bags_models[bag][k].fit(train_x, train_y[:,k])

    def predict_proba(self, test_x):
        for bag in range(self.bags):
            for k in range(self.nb_classes):
                predict = self.bags_models[bag][k].predict(test_x)
                predict = predict.reshape((predict.shape[0], 1))

                # concatenate predictions
                if k==0:
                    test_predicted = predict
                else:
                    test_predicted = np.concatenate((test_predicted, predict), axis=1)

            # limitate predictions to [0,1]
            test_predicted = np.apply_along_axis(lambda x:map( lambda y:max(y,0),x), 0, test_predicted)
            test_predicted = np.apply_along_axis(lambda x:map( lambda y:min(y,1),x), 0, test_predicted)

            # normalize class probabilities
            test_predicted = np.apply_along_axis(lambda x:x/sum(x), 1, test_predicted)

            if bag == 0:
                total_predicted = test_predicted
            else:
                total_predicted = total_predicted + test_predicted

        total_predicted = total_predicted / self.bags

        return test_predicted


class PMC_RandomForest(): # Probabilistic MultiClassification
    def __init__(self, nb_classes, bags=1, param={}):
        # libraries
        from sklearn.ensemble import RandomForestRegressor

        # params
        self.n_estimators = param.get('n_estimators',10)
        self.max_features = param.get('max_features',None)
        self.min_samples_leaf = param.get('min_samples_leaf',5)
        self.bootstrap = param.get('bootstrap',True)
        self.n_jobs = param.get('n_jobs',-1)
        self.random_state = param.get('random_state',0)
        #common
        self.nb_classes = nb_classes
        self.bags = bags
        self.bags_models = tuple()
        self.train_y = None
        for bag in range(self.bags):
            models = tuple()
            for k in range(self.nb_classes):
                model = RandomForestRegressor(n_estimators = self.n_estimators, max_features=self.max_features,
                                            min_samples_leaf = self.min_samples_leaf, bootstrap = self.bootstrap,
                                            n_jobs = self.n_jobs, random_state  = self.random_state + bag)
                models = models + (model,)
            self.bags_models = self.bags_models + (models, )

    def fit(self, train_x, train_y, sample_weight = None):
        self.train_y = train_y
        for bag in range(self.bags):
            for k in range(self.nb_classes):
                self.bags_models[bag][k].fit(train_x, train_y[:,k], sample_weight)

    def predict_proba(self, test_x):
        for bag in range(self.bags):
            for k in range(self.nb_classes):
                predict = self.bags_models[bag][k].predict(test_x)
                predict = predict.reshape((predict.shape[0], 1))

                # concatenate predictions
                if k==0:
                    test_predicted = predict
                else:
                    test_predicted = np.concatenate((test_predicted, predict), axis=1)

            # limitate predictions to [0,1]
            test_predicted = np.apply_along_axis(lambda x:map( lambda y:max(y,0),x), 0, test_predicted)
            test_predicted = np.apply_along_axis(lambda x:map( lambda y:min(y,1),x), 0, test_predicted)

            # normalize class probabilities
            test_predicted = np.apply_along_axis(lambda x:x/sum(x), 1, test_predicted)

            if bag == 0:
                total_predicted = test_predicted
            else:
                total_predicted = total_predicted + test_predicted

        total_predicted = total_predicted / self.bags

        return test_predicted


class PMC_MultiTaskRandomForest(): # Probabilistic MultiClassification

    def __init__(self, nb_classes, bags=1, params={}):

        # libraries
        from sklearn.ensemble import RandomForestRegressor

        # params
        self.params = params

        #common
        self.nb_classes = nb_classes
        self.bags = bags
        self.bags_models = tuple()
        self.train_y = None

        for bag in range(self.bags):
            model = RandomForestRegressor()
            model.set_params(**self.params)
            model.set_params(random_state = (self.params['random_state'] + bag))
            self.bags_models = self.bags_models + (model, )

    def fit(self, train_x, train_y, sample_weight = None):
        self.train_y = train_y
        for bag in range(self.bags):
            self.bags_models[bag].fit(train_x, train_y, sample_weight)

    def predict_proba(self, test_x):
        for bag in range(self.bags):
            test_predicted = self.bags_models[bag].predict(test_x)

            # limitate predictions to [0,1]
            test_predicted = np.apply_along_axis(lambda x:map( lambda y:max(y,0),x), 0, test_predicted)
            test_predicted = np.apply_along_axis(lambda x:map( lambda y:min(y,1),x), 0, test_predicted)

            # normalize class probabilities
            test_predicted = np.apply_along_axis(lambda x:x/sum(x), 1, test_predicted)

            if bag == 0:
                total_predicted = test_predicted
            else:
                total_predicted = total_predicted + test_predicted

        total_predicted = total_predicted / self.bags

        return test_predicted


class ProbMC_XGBoost(): # Probabilistic MultiClassification
    def __init__(self, nb_classes, bags=1, param={}):
        import xgboost as xgb
        from xgboost.sklearn import XGBRegressor

        self.nb_classes = nb_classes
        self.objective = param.get('objective','reg:linear')
        self.nthread = param.get('nthread',-1)
        self.n_estimators = param.get('n_estimators',10)
        self.max_depth = param.get('max_depth', 6)
        self.learning_rate = param.get('learning_rate', 0.3)
        self.colsample_bytree = param.get('colsample_bytree', 1.0)
        self.subsample = param.get('subsample', 1.0)
        self.missing = param.get('missing', None)
        self.seed = param.get('seed', 0)
        self.bags = bags
        self.bags_models = tuple()
        self.train_y = None
        for bag in range(self.bags):
            models = tuple()
            for k in range(self.nb_classes):
                model = XGBRegressor(objective = self.objective, nthread = self.nthread, seed = self.seed + bag,
                                     n_estimators = self.n_estimators, missing = self.missing,
                                     max_depth = self.max_depth, learning_rate = self.learning_rate,
                                     colsample_bytree = self.colsample_bytree, subsample = self.subsample)
                models = models + (model,)
            self.bags_models = self.bags_models + (models, )

    def fit(self, train_x, train_y):
        self.train_y = train_y
        for bag in range(self.bags):
            for k in range(self.nb_classes):
                self.bags_models[bag][k].fit(train_x, train_y[:,k])

    def predict_proba(self, test_x):
        for bag in range(self.bags):
            for k in range(self.nb_classes):
                predict = self.bags_models[bag][k].predict(test_x)
                predict = predict.reshape((predict.shape[0], 1))

                # concatenate predictions
                if k==0:
                    test_predicted = predict
                else:
                    test_predicted = np.concatenate((test_predicted, predict), axis=1)

            # limitate predictions to [0,1]
            test_predicted = np.apply_along_axis(lambda x:map( lambda y:max(y,0),x), 0, test_predicted)
            test_predicted = np.apply_along_axis(lambda x:map( lambda y:min(y,1),x), 0, test_predicted)

            # normalize class probabilities
            test_predicted = np.apply_along_axis(lambda x:x/sum(x), 1, test_predicted)

            if bag == 0:
                total_predicted = test_predicted
            else:
                total_predicted = total_predicted + test_predicted

        total_predicted = total_predicted / self.bags

        return test_predicted
