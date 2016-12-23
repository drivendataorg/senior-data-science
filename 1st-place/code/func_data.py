# -*- coding: utf-8 -*-
"""
Created on Wed Jul  6 07:09:18 2016

"""

# For number crunching
import numpy as np
import pandas as pd

# For preprocessing
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import StandardScaler

# Misc
import json
import os


def load_sequence(file_id, columns_id=['columns'], params={}, target_id='default'):

    # Get Parameters
    add_meta           = params.get('add_meta', False)
    imputer_strategy   = params.get('imputer_strategy', None)

    # columns_id is a list of filenames to read and concatenate
    filename = str(file_id).zfill(5)

    df = []

    for column_id in columns_id:
        i_df = pd.read_csv('../preprocessed_data/train/{}/{}.csv'.format(filename, column_id)).values
        df.append(i_df)

    df = np.column_stack(df)

    if add_meta:
        meta = json.load(open(os.path.join('../public_data', 'train', filename, 'meta.json')))
        df = np.append(df, np.repeat(len(meta['annotators']), df.shape[0]).reshape(df.shape[0],1), axis=1)

    if imputer_strategy is not None:
        df = np.vstack((df, np.repeat(-99999, df.shape[1]))) # usefull to control al NAs columns
        data_imputer = Imputer(strategy=imputer_strategy)
        df = data_imputer.fit_transform(df)
        df = np.delete(df, (df.shape[0]-1), axis=0) # delete the last row, added before
        df[df == -99999] = np.nan

    if target_id == 'default':
        target = np.asarray(pd.read_csv('../public_data/train/{}/targets.csv'.format(filename)))[:, 2:]
    else:
        target = np.asarray(pd.read_csv('../preprocessed_data/train/{}/{}.csv'.format(filename, target_id)))[:, 2:]

    return df, target

#train_x, train_y = load_sequence(1,add_meta=True, imputer_strategy='mean')


def load_sequences(file_ids, columns_id=['columns'], params={}, target_id='default'):

    # Get Parameters
    add_meta           = params.get('add_meta', False)
    imputer_strategy   = params.get('imputer_strategy', None)

    # Set Parameters
    load_sequence_params = {'add_meta':add_meta, 'imputer_strategy':imputer_strategy}

    # Main
    x_es = []
    y_es = []
    seq_es = []

    for file_id in file_ids:
        data, target = load_sequence(file_id, columns_id, params=load_sequence_params, target_id=target_id)

        x_es.append(data)
        y_es.append(target)
        seq_es.append(np.repeat(file_id, target.shape[0]))

    return np.row_stack(x_es), np.row_stack(y_es), np.concatenate(seq_es)

#sequence_train = [1,2,3,4,5,6,7,8,9,10]
#data = ['columns', 'columns_v1', 'columns_v2']
#train_x, train_y, train_seq = load_sequences(sequence_train, data, add_meta=True, imputer_strategy='mean')


def load_test(columns_id=['columns'], params={}):

    # Get Parameters
    add_meta           = params.get('add_meta', False)
    default_annotators = params.get('default_annotators', 2)
    imputer_strategy   = params.get('imputer_strategy', None)


    rows = []
    test_x = []
    df= []

    for te_ind_str in sorted(os.listdir(os.path.join('../public_data', 'test'))):
        te_ind = int(te_ind_str)

        idf = []

        for column_id in columns_id:
            i_df = pd.read_csv('../preprocessed_data/test/{}/{}.csv'.format(te_ind_str, column_id)).values
            idf.append(i_df)

        df = np.column_stack(idf)

        meta = json.load(open(os.path.join('../public_data', 'test', te_ind_str, 'meta.json')))

        if add_meta:
            annotators = len(meta['annotators'])
            if annotators == 0:
                annotators = default_annotators
            df = np.append(df, np.repeat(annotators, df.shape[0]).reshape(df.shape[0],1), axis=1)

        if imputer_strategy is not None:
            df = np.vstack((df, np.repeat(-99999, df.shape[1]))) # usefull to control al NAs columns
            data_imputer = Imputer(strategy=imputer_strategy)
            df = data_imputer.fit_transform(df)
            df = np.delete(df, (df.shape[0]-1), axis=0) # delete the last row, added before
            df[df == -99999] = np.nan

        starts = range(meta['end'])
        ends = range(1, meta['end'] + 1)

        for start, end, feature in zip(starts, ends, df):
            rows.append([te_ind, start, end])
            test_x.append(feature.tolist())

    rows = np.asarray(rows)
    test_x = np.asarray(test_x)

    return (rows, test_x)

#rows, test_x = load_test()


def load_target_location(file_ids):

    # Main
    y_es = []

    for file_id in file_ids:

        # columns_id is a list of filenames to read and concatenate
        filename = str(file_id).zfill(5)

        target = np.asarray(pd.read_csv('../preprocessed_data/train/{}/targets_locations.csv'.format(filename)))[:, 2:]

        y_es.append(target)

    return np.row_stack(y_es)


def load_seq(file_ids):
    x_es = []
    y_es = []

    for file_id in file_ids:

        filename = str(file_id).zfill(5)
        df = pd.read_csv('../public_data/train/{}/columns.csv'.format(filename))
        data = df.values
        target = np.asarray(pd.read_csv('../public_data/train/{}/targets.csv'.format(filename)))[:, 2:]

        x_es.append(data)
        y_es.append(target)

    return np.row_stack(x_es), np.row_stack(y_es)


def get_clean_sequences(file_ids):
    for i in file_ids:
        train_x, train_y = load_seq([i])
        train_y_has_annotation = np.isfinite(train_y.sum(1))
        train_x = train_x[train_y_has_annotation]
        if i==1:
            train_seq = np.array([i]).repeat(train_x.shape[0])
        else:
            train_seq = np.concatenate((train_seq, np.array([i]).repeat(train_x.shape[0])), axis=0)
    return train_seq


def load_train_y(file_ids):

    # Main
    y_es = []

    for file_id in file_ids:
        filename = str(file_id).zfill(5)
        target = np.asarray(pd.read_csv('../public_data/train/{}/targets.csv'.format(filename)))[:, 2:]

        y_es.append(target)

    return np.row_stack(y_es)


def load_scores(files_txt, directory='submissions'):
    for i,file_txt in enumerate(files_txt):
            df = pd.read_csv('../{}/{}_score.csv'.format(directory, file_txt), header=None)
            df.columns = [file_txt]
            df.loc[df.shape[0]] = df.mean()
            if i==0:
                df_result = df
            else:
                df_result = pd.concat((df_result, df), axis=1)
    return df_result


def load_submissions(files_txt, directory='submissions'):
    for i,file_txt in enumerate(files_txt):
            df = pd.read_csv('../{}/{}_submission.csv'.format(directory, file_txt), header=0)
            df.columns = [file_txt + '|'] + df.columns.values
            df = df.iloc[:,3:23]
            if i==0:
                df_result = df
            else:
                df_result = pd.concat((df_result, df), axis=1)
    return df_result.values


def load_L1_train(files_txt, directory='submissions'):
    for i,file_txt in enumerate(files_txt):
            df = pd.read_csv('../{}/{}_valid.csv'.format(directory, file_txt), header=0)
            df.columns = [file_txt + '|'] + df.columns.values
            if i==0:
                df_result = df
            else:
                df_result = pd.concat((df_result, df), axis=1)
    return df_result.values


def whole_preprocess(train_x, train_y, train_seq, rows, test_x, params={}):

    # Get Parameters
    remove_nan_targets = params.get('remove_nan_targets', False)
    missing = params.get('missing', None)
    imputer_strategy = params.get('imputer_strategy', None)
    float32 = params.get('float32', False)
    scale = params.get('scale', False)

    # Remove_nan_targets
    if remove_nan_targets:
        train_y_has_annotation = np.isfinite(train_y.sum(1))
        train_x = train_x[train_y_has_annotation]
        train_y = train_y[train_y_has_annotation]
        train_seq = train_seq[train_y_has_annotation]

    # Missing data
    if missing is not None:
        train_x[np.isnan(train_x)] = missing
        test_x[np.isnan(test_x)] = missing

    # Imputation
    if imputer_strategy is not None:
        imputer = Imputer(strategy=imputer_strategy)
        imputer.fit(train_x)
        train_x = imputer.transform(train_x)
        test_x = imputer.transform(test_x)

    # Change type to float32 (needed for NN with Keras)
    if float32:
        train_x = train_x.astype(np.float32)
        test_x = test_x.astype(np.float32)
        train_y = train_y.astype(np.float32)

    # Scale data
    if scale:
        scaler = StandardScaler().fit(train_x)
        train_x = scaler.transform(train_x)
        test_x = scaler.transform(test_x)

    return (train_x, train_y, train_seq, rows, test_x)


def batch_preprocess(train_x, train_y, test_x, params={}):

    # Get Parameters
    missing = params.get('missing', None)
    imputer_strategy = params.get('imputer_strategy', None)
    float32 = params.get('float32', False)
    scale = params.get('scale', False)

    # Missing data
    if missing is not None:
        train_x[np.isnan(train_x)] = missing
        test_x[np.isnan(test_x)] = missing

    # Imputation
    if imputer_strategy is not None:
        imputer = Imputer(strategy=imputer_strategy)
        imputer.fit(train_x)
        train_x = imputer.transform(train_x)
        test_x = imputer.transform(test_x)

    # Change type to float32 (needed for NN with Keras)
    if float32:
        train_x = train_x.astype(np.float32)
        test_x = test_x.astype(np.float32)
        train_y = train_y.astype(np.float32)

    # Scale data
    if scale:
        scaler = StandardScaler().fit(train_x)
        train_x = scaler.transform(train_x)
        test_x = scaler.transform(test_x)

    return (train_x, train_y, test_x)


def get_past_data(data_table, seq, n_past=1, nan=np.nan, noise=None):
    '''
    data_table = 2d numpy array
    '''
    result = []
    for s in np.unique(seq):
        df = data_table[seq==s,:]
        if noise is None:
            for i in range(n_past):
                df = np.r_[np.repeat(nan, data_table.shape[1]).reshape((1,data_table.shape[1])), df]
            df = df[:-n_past,:]
            result.append(df)
        else:
            mini_seq = []
            for x in range(df.shape[0]):
                mini_seq.append(int(x/noise))
            for ss in np.unique(mini_seq):
                ddf = df[mini_seq == ss, :]
                for i in range(n_past):
                    ddf = np.r_[np.repeat(nan, data_table.shape[1]).reshape((1,data_table.shape[1])), ddf]
                ddf = ddf[:-n_past,:]
                result.append(ddf)

    result = np.vstack(result)

    return result


def get_future_data(data_table, seq, n_fut=1, nan=np.nan, noise=None):
    '''
    data_table = 2d numpy array
    '''
    result = []
    for s in np.unique(seq):
        df = data_table[seq==s,:]
        if noise is None:
            for i in range(n_fut):
                df = np.r_[df, np.repeat(nan, data_table.shape[1]).reshape((1,data_table.shape[1]))]
            df = df[n_fut:,:]
            result.append(df)
        else:
            mini_seq = []
            for x in range(df.shape[0]):
                mini_seq.append(int(x/noise))
            for ss in np.unique(mini_seq):
                ddf = df[mini_seq == ss, :]
                for i in range(n_fut):
                    ddf = np.r_[ddf, np.repeat(nan, data_table.shape[1]).reshape((1,data_table.shape[1]))]
                ddf = ddf[n_fut:,:]
                result.append(ddf)
    result = np.vstack(result)

    return result
