# -*- coding: utf-8 -*-

# For number crunching
import numpy as np
import pandas as pd

# Misc
import os

# Functions libraries
import func_visualise_data as fvd

import warnings
warnings.filterwarnings('ignore')

# Define functions

def extract_rssi_v1A(df, n_split):
    '''
    This function extract features from collected 'rssi' data.
    df: pandas dataFrame of data for a given time-window (1s)
    n_split: int[1,] number of groups inside the given time-window
    return: numpy 2D-array of new features:
        [nanmean] of [Kitchen_AP, Lounge_AP, Upstairs_AP, Study_AP] 
        for each 'n_splits' group
    '''
    num_ff = 4 * n_split  # Number of final features

    if df.shape[0] == 0:
        
        # No data, assign NAs
        result_row = np.repeat(np.nan, num_ff)
    else:
        
        # feature engineering
        df = -df
        
        # define row_id
        row_id = range(0,df.shape[0])
        
        # split the data in n_split groups
        splits = []
        tmp_row_id = row_id
        i_prev_split = []
        for i in range(n_split,0,-1):
            i_split = [tmp_row_id[:int(round(len(tmp_row_id)/float(i)))]]
            if len(i_split[0]) == 0:
                i_split = i_prev_split
            i_prev_split = i_split
            splits = splits + i_split
            tmp_row_id = tmp_row_id[int(round(len(tmp_row_id)/float(i))):]
        
        # Apply functions for each group
        result_row = np.array([])
        for i1, split in enumerate(splits):
            tmp_df = df.iloc[split,0:4].copy()
            result_row = np.append(result_row, tmp_df.apply(lambda x: round(np.mean(x),0), axis=0).values)

    return result_row


def extract_rssi_v1B(df, n_split):
    '''
    This function extract features from collected 'rssi' data.
    df: pandas dataFrame of data for a given time-window (1s)
    n_split: int[1,] number of groups inside the given time-window
    return: numpy 2D-array of new features:
        [nanmean] of 4 'rssi' values
        [nanmax, nanmin] of [Kitchen_AP, Lounge_AP, Upstairs_AP, Study_AP] 
        for each 'n_splits' group
    '''
    num_ff = 9 * n_split  # Number of final features

    if df.shape[0] == 0:
        
        # No data, assign NAs
        result_row = np.repeat(np.nan, num_ff)
    else:
        
        # feature engineering
        df = -df
        
        # define row_id
        row_id = range(0,df.shape[0])
        
        # split the data in n_split groups
        splits = []
        tmp_row_id = row_id
        i_prev_split = []
        for i in range(n_split,0,-1):
            i_split = [tmp_row_id[:int(round(len(tmp_row_id)/float(i)))]]
            if len(i_split[0]) == 0:
                i_split = i_prev_split
            i_prev_split = i_split
            splits = splits + i_split
            tmp_row_id = tmp_row_id[int(round(len(tmp_row_id)/float(i))):]
        
        # Apply functions for each group
        result_row = np.array([])
        for i1, split in enumerate(splits):
            tmp_df =  df.iloc[split,0:4].copy().apply(lambda x: round(np.mean(x),0), axis=0).values
            result_row = np.append(result_row, round(np.nansum(tmp_df),0))
            result_row = np.append(result_row, ((np.nanargmax(tmp_df)+1)==1).astype(int))
            result_row = np.append(result_row, ((np.nanargmax(tmp_df)+1)==2).astype(int))
            result_row = np.append(result_row, ((np.nanargmax(tmp_df)+1)==3).astype(int))
            result_row = np.append(result_row, ((np.nanargmax(tmp_df)+1)==4).astype(int))
            result_row = np.append(result_row, ((np.nanargmin(tmp_df)+1)==1).astype(int))
            result_row = np.append(result_row, ((np.nanargmin(tmp_df)+1)==2).astype(int))
            result_row = np.append(result_row, ((np.nanargmin(tmp_df)+1)==3).astype(int))
            result_row = np.append(result_row, ((np.nanargmin(tmp_df)+1)==4).astype(int))

    return result_row


def extract_rssi_v1C(df, n_split):
    '''
    This function extract features from collected 'rssi' data.
    df: pandas dataFrame of data for a given time-window (1s)
    n_split: int[1,] number of groups inside the given time-window
    return: numpy 2D-array of new features:
        [mean] of [Kitchen_AP, Lounge_AP, Upstairs_AP, Study_AP] 
        for each 'n_splits' group
    '''
    num_ff = 4 * n_split  # Number of final features

    if df.shape[0] == 0:
        
        # No data, assign NAs
        result_row = np.repeat(np.nan, num_ff)
    else:
        
        # feature engineering
        df = -df
        
        # define row_id
        row_id = range(0,df.shape[0])
        
        # split the data in n_split groups
        splits = []
        tmp_row_id = row_id
        i_prev_split = []
        for i in range(n_split,0,-1):
            i_split = [tmp_row_id[:int(round(len(tmp_row_id)/float(i)))]]
            if len(i_split[0]) == 0:
                i_split = i_prev_split
            i_prev_split = i_split
            splits = splits + i_split
            tmp_row_id = tmp_row_id[int(round(len(tmp_row_id)/float(i))):]
        
        # Apply functions for each group
        result_row = np.array([])
        for i1, split in enumerate(splits):
            tmp_df = df.iloc[split,0:4].copy()
            tmp_df[np.isnan(tmp_df)] = 0
            result_row = np.append(result_row, tmp_df.apply(lambda x: round(np.mean(x),0), axis=0).values)

    return result_row


def extract_rssi_v1D(df, n_split):
    '''
    This function extract features from collected 'rssi' data.
    df: pandas dataFrame of data for a given time-window (1s)
    n_split: int[1,] number of groups inside the given time-window
    return: numpy 2D-array of new features:
        [mean] of 4 'rssi' values
        [max, min] of [Kitchen_AP, Lounge_AP, Upstairs_AP, Study_AP] 
        for each 'n_splits' group
    '''
    num_ff = 9 * n_split  # Number of final features

    if df.shape[0] == 0:
        
        # No data, assign NAs
        result_row = np.repeat(np.nan, num_ff)
    else:
        
        # feature engineering
        df = -df
        
        # define row_id
        row_id = range(0,df.shape[0])
        
        # split the data in n_split groups
        splits = []
        tmp_row_id = row_id
        i_prev_split = []
        for i in range(n_split,0,-1):
            i_split = [tmp_row_id[:int(round(len(tmp_row_id)/float(i)))]]
            if len(i_split[0]) == 0:
                i_split = i_prev_split
            i_prev_split = i_split
            splits = splits + i_split
            tmp_row_id = tmp_row_id[int(round(len(tmp_row_id)/float(i))):]
        
        # Apply functions for each group
        result_row = np.array([])
        for i1, split in enumerate(splits):
            tmp_df = df.iloc[split, 0:4].copy()
            tmp_df[np.isnan(tmp_df)] = 0
            tmp_df = tmp_df.apply(lambda x: round(np.mean(x),0), axis=0).values
            tmp_df[tmp_df==0] = np.nan
            result_row = np.append(result_row, round(np.nansum(tmp_df),0))
            result_row = np.append(result_row, ((np.nanargmax(tmp_df)+1)==1).astype(int))
            result_row = np.append(result_row, ((np.nanargmax(tmp_df)+1)==2).astype(int))
            result_row = np.append(result_row, ((np.nanargmax(tmp_df)+1)==3).astype(int))
            result_row = np.append(result_row, ((np.nanargmax(tmp_df)+1)==4).astype(int))
            result_row = np.append(result_row, ((np.nanargmin(tmp_df)+1)==1).astype(int))
            result_row = np.append(result_row, ((np.nanargmin(tmp_df)+1)==2).astype(int))
            result_row = np.append(result_row, ((np.nanargmin(tmp_df)+1)==3).astype(int))
            result_row = np.append(result_row, ((np.nanargmin(tmp_df)+1)==4).astype(int))

    return result_row


def extract_rssi_v1E(df, n_split):
    '''
    This function extract features from collected 'rssi' data.
    df: pandas dataFrame of data for a given time-window (1s)
    n_split: int[1,] number of groups inside the given time-window
    return: numpy 2D-array of new features:
        [meansdiff] as 'nanmean' - 'mean' (see B & D extraction functions)
        for each 'n_splits' group
    '''
    num_ff = 1 * n_split  # Number of final features

    if df.shape[0] == 0:
        
        # No data, assign NAs
        result_row = np.repeat(np.nan, num_ff)
    else:
        
        # feature engineering
        df = -df
        
        # define row_id
        row_id = range(0,df.shape[0])
        
        # split the data in n_split groups
        splits = []
        tmp_row_id = row_id
        i_prev_split = []
        for i in range(n_split,0,-1):
            i_split = [tmp_row_id[:int(round(len(tmp_row_id)/float(i)))]]
            if len(i_split[0]) == 0:
                i_split = i_prev_split
            i_prev_split = i_split
            splits = splits + i_split
            tmp_row_id = tmp_row_id[int(round(len(tmp_row_id)/float(i))):]
        
        # Apply functions for each group
        result_row = np.array([])
        for i1, split in enumerate(splits):
            tmp_df = df.iloc[split,0:4].copy()
            tmp_df =  tmp_df.apply(lambda x: round(np.mean(x),0), axis=0).values  
            tmp_memory = [round(np.nansum(tmp_df),0)]
            
            tmp_df = df.iloc[split,0:4].copy()
            tmp_df[np.isnan(tmp_df)] = 0
            tmp_df =  tmp_df.apply(lambda x: round(np.mean(x),0), axis=0).values 
            tmp_memory.append(round(np.nansum(tmp_df),0))

            result_row = np.append(result_row, (tmp_memory[0]-tmp_memory[1]))

    return result_row


def extract_rssi_v1F(df, n_split):
    '''
    This function extract features from collected 'rssi' data.
    df: pandas dataFrame of data for a given time-window (1s)
    n_split: int[1,] number of groups inside the given time-window
    return: numpy 2D-array of new features:
        [means] of [maxKitchen_AP, maxLounge_AP, maxUpstairs_AP, maxStudy_AP,
                    minKitchen_AP, minLounge_AP, minUpstairs_AP, minStudy_AP]
        for each 'n_splits' group
    '''
    num_ff = 8 * n_split  # Number of final features

    if df.shape[0] == 0:
        
        # No data, assign NAs
        result_row = np.repeat(np.nan, num_ff)
    else:
        
        # feature engineering
        df = -df
        df['max_Kitchen_AP'] = (df[['Kitchen_AP','Lounge_AP','Upstairs_AP',
                          'Study_AP']].apply(lambda x: np.nanargmax(abs(x.values)) + 1,axis=1) == 1).astype(int)
        df['max_Lounge_AP'] = (df[['Kitchen_AP','Lounge_AP','Upstairs_AP',
                          'Study_AP']].apply(lambda x: np.nanargmax(abs(x.values)) + 1,axis=1) == 2).astype(int)
        df['max_Upstairs_AP'] = (df[['Kitchen_AP','Lounge_AP','Upstairs_AP',
                          'Study_AP']].apply(lambda x: np.nanargmax(abs(x.values)) + 1,axis=1) == 3).astype(int)
        df['max_Study_AP'] = (df[['Kitchen_AP','Lounge_AP','Upstairs_AP',
                          'Study_AP']].apply(lambda x: np.nanargmax(abs(x.values)) + 1,axis=1) == 4).astype(int)
        df['min_Kitchen_AP'] = (df[['Kitchen_AP','Lounge_AP','Upstairs_AP',
                          'Study_AP']].apply(lambda x: np.nanargmin(abs(x.values)) + 1,axis=1) == 1).astype(int)
        df['min_Lounge_AP'] = (df[['Kitchen_AP','Lounge_AP','Upstairs_AP',
                          'Study_AP']].apply(lambda x: np.nanargmin(abs(x.values)) + 1,axis=1) == 2).astype(int)
        df['min_Upstairs_AP'] = (df[['Kitchen_AP','Lounge_AP','Upstairs_AP',
                          'Study_AP']].apply(lambda x: np.nanargmin(abs(x.values)) + 1,axis=1) == 3).astype(int)
        df['min_Study_AP'] = (df[['Kitchen_AP','Lounge_AP','Upstairs_AP',
                          'Study_AP']].apply(lambda x: np.nanargmin(abs(x.values)) + 1,axis=1) == 4).astype(int)
        
        # define row_id
        row_id = range(0,df.shape[0])
        
        # split the data in n_split groups
        splits = []
        tmp_row_id = row_id
        i_prev_split = []
        for i in range(n_split,0,-1):
            i_split = [tmp_row_id[:int(round(len(tmp_row_id)/float(i)))]]
            if len(i_split[0]) == 0:
                i_split = i_prev_split
            i_prev_split = i_split
            splits = splits + i_split
            tmp_row_id = tmp_row_id[int(round(len(tmp_row_id)/float(i))):]
        
        # Apply functions for each group
        result_row = np.array([])
        for i1, split in enumerate(splits):
            tmp_df = df.iloc[split, 4:12].copy()
            tmp_df[np.isnan(tmp_df)] = 0
            result_row = np.append(result_row, tmp_df.apply(lambda x: np.mean(x), axis=0).values)

    return result_row


# Column nomes
series_name = 'rssi_M04'

column_names_A_s1 = []
for i1 in range(1,(1+1)):
    for i2 in ['Kitchen_AP','Lounge_AP','Upstairs_AP','Study_AP']:
        for i3 in ['nanmean']:
            column_names_A_s1.append('{0}_T{1}of{2}_{3}_{4}'.format(series_name, i1+1, 1, i2, i3))

column_names_B_s1 = []       
for i1 in range(1,(1+1)):
    column_names_B_s1.append('{0}_T{1}of{2}_{3}'.format(series_name, i1+1, 1, 'nanmean'))
    for i2 in ['Kitchen_AP','Lounge_AP','Upstairs_AP','Study_AP']:
        for i3 in ['nanmax','nanmin']:
            column_names_B_s1.append('{0}_T{1}of{2}_{3}_{4}'.format(series_name, i1+1, 1, i2, i3))   

column_names_C_s1 = []
for i1 in range(1,(1+1)):
    for i2 in ['Kitchen_AP','Lounge_AP','Upstairs_AP','Study_AP']:
        for i3 in ['mean']:
            column_names_C_s1.append('{0}_T{1}of{2}_{3}_{4}'.format(series_name, i1+1, 1, i2, i3))

column_names_D_s1 = []       
for i1 in range(1,(1+1)):
    column_names_D_s1.append('{0}_T{1}of{2}_{3}'.format(series_name, i1+1, 1, 'mean'))
    for i2 in ['Kitchen_AP','Lounge_AP','Upstairs_AP','Study_AP']:
        for i3 in ['max','min']:
            column_names_D_s1.append('{0}_T{1}of{2}_{3}_{4}'.format(series_name, i1+1, 1, i2, i3))  

column_names_E_s1 = []       
for i1 in range(1,(1+1)):
    column_names_E_s1.append('{0}_T{1}of{2}_{3}'.format(series_name, i1+1, 1, 'meansdiff'))

column_names_F_s1 = [] 
for i1 in range(1,(1+1)):
    for i2 in ['maxKitchen_AP','maxLounge_AP','maxUpstairs_AP','maxStudy_AP',
               'minKitchen_AP','minLounge_AP','minUpstairs_AP','minStudy_AP']:
        for i3 in ['mean']:
            column_names_F_s1.append('{0}_T{1}of{2}_{3}_{4}'.format(series_name, i1+1, 1, i2, i3))

column_names_A_s2 = []
for i1 in range(1,(1+2)):
    for i2 in ['Kitchen_AP','Lounge_AP','Upstairs_AP','Study_AP']:
        for i3 in ['nanmean']:
            column_names_A_s2.append('{0}_T{1}of{2}_{3}_{4}'.format(series_name, i1+1, 2, i2, i3))

column_names_B_s2 = []       
for i1 in range(1,(1+2)):
    column_names_B_s2.append('{0}_T{1}of{2}_{3}'.format(series_name, i1+1, 2, 'nanmean'))
    for i2 in ['Kitchen_AP','Lounge_AP','Upstairs_AP','Study_AP']:
        for i3 in ['nanmax','nanmin']:
            column_names_B_s2.append('{0}_T{1}of{2}_{3}_{4}'.format(series_name, i1+1, 2, i2, i3))   

column_names_C_s2 = []
for i1 in range(1,(1+2)):
    for i2 in ['Kitchen_AP','Lounge_AP','Upstairs_AP','Study_AP']:
        for i3 in ['mean']:
            column_names_C_s2.append('{0}_T{1}of{2}_{3}_{4}'.format(series_name, i1+1, 2, i2, i3))

column_names_D_s2 = []       
for i1 in range(1,(1+2)):
    column_names_D_s2.append('{0}_T{1}of{2}_{3}'.format(series_name, i1+1, 2, 'mean'))
    for i2 in ['Kitchen_AP','Lounge_AP','Upstairs_AP','Study_AP']:
        for i3 in ['max','min']:
            column_names_D_s2.append('{0}_T{1}of{2}_{3}_{4}'.format(series_name, i1+1, 2, i2, i3))  

column_names_E_s2 = []       
for i1 in range(1,(1+2)):
    column_names_E_s2.append('{0}_T{1}of{2}_{3}'.format(series_name, i1+1, 2, 'meansdiff'))

column_names_F_s2 = [] 
for i1 in range(1,(1+2)):
    for i2 in ['maxKitchen_AP','maxLounge_AP','maxUpstairs_AP','maxStudy_AP',
               'minKitchen_AP','minLounge_AP','minUpstairs_AP','minStudy_AP']:
        for i3 in ['mean']:
            column_names_F_s2.append('{0}_T{1}of{2}_{3}_{4}'.format(series_name, i1+1, 2, i2, i3))


"""
Iterate over all training/testing directories
"""

for train_test in ('train', 'test', ): 
    
    # Print msg 
    if train_test is 'train': 
        print ('Extracting features from training data.\n')
    else: 
        print ('\n\n\nExtracting features from testing data.\n')
    
    # iterate folders
    for fi, file_id in enumerate(sorted(os.listdir('../public_data/{}/'.format(train_test)))):
        stub_name = str(file_id).zfill(5)

        if train_test == 'train' or np.mod(fi, 50) == 0:
             print ("Starting feature extraction for {}/{}".format(train_test, stub_name))
        
        
        #### Generate ds_rssi_M00_v1A/B/C/D/E/F_s1 / ds_rssi_M00_v1A/B/C/D/E/F_s2 (12 files)

        # Use the sequence loader to load the data from the directory. 
        data = fvd.Sequence('../public_data', '../public_data/{}/{}'.format(train_test, stub_name))
        data.load()
        data.rssi_movSum(4,4) # Transform rssi data to mov_sum
        
        # iterate time-window data
        rows_A_s1 = []
        rows_A_s2 = []
        rows_B_s1 = []
        rows_B_s2 = []
        rows_C_s1 = []
        rows_C_s2 = []
        rows_D_s1 = []
        rows_D_s2 = []
        rows_E_s1 = []
        rows_E_s2 = []
        rows_F_s1 = []
        rows_F_s2 = []
        for ri, (lu, (accel, rssi, pir, vid_lr, vid_k, vid_h)) in enumerate(data.iterate()):
            
            # Append the row to the full set of features
            rows_A_s1.append(extract_rssi_v1A(rssi, 1))
            rows_A_s2.append(extract_rssi_v1A(rssi, 2))
            rows_B_s1.append(extract_rssi_v1B(rssi, 1))
            rows_B_s2.append(extract_rssi_v1B(rssi, 2))
            rows_C_s1.append(extract_rssi_v1C(rssi, 1))
            rows_C_s2.append(extract_rssi_v1C(rssi, 2))
            rows_D_s1.append(extract_rssi_v1D(rssi, 1))
            rows_D_s2.append(extract_rssi_v1D(rssi, 2))
            rows_E_s1.append(extract_rssi_v1E(rssi, 1))
            rows_E_s2.append(extract_rssi_v1E(rssi, 2))
            rows_F_s1.append(extract_rssi_v1F(rssi, 1))
            rows_F_s2.append(extract_rssi_v1F(rssi, 2))
            
            # Report progress 
            if train_test is 'train':
                if np.mod(ri + 1, 50) == 0:
                    print ("{:5}".format(str(ri + 1))),

                if np.mod(ri + 1, 500) == 0:
                    print
        
        # Wrap data
        rows_A_s1 = np.vstack(rows_A_s1)
        rows_A_s2 = np.vstack(rows_A_s2)
        rows_B_s1 = np.vstack(rows_B_s1)
        rows_B_s2 = np.vstack(rows_B_s2)
        rows_C_s1 = np.vstack(rows_C_s1)
        rows_C_s2 = np.vstack(rows_C_s2)
        rows_D_s1 = np.vstack(rows_D_s1)
        rows_D_s2 = np.vstack(rows_D_s2)
        rows_E_s1 = np.vstack(rows_E_s1)
        rows_E_s2 = np.vstack(rows_E_s2)
        rows_F_s1 = np.vstack(rows_F_s1)
        rows_F_s2 = np.vstack(rows_F_s2)

        # save data
        directory = '../preprocessed_data/{}/{}/'.format(train_test, stub_name)
        if not os.path.exists(directory):
            os.makedirs(directory)
        
        df_A_s1 = pd.DataFrame(rows_A_s1)
        df_A_s1.columns = column_names_A_s1
        df_A_s1.to_csv('{}ds_rssi_M04_v1A_s1.csv'.format(directory), index=False)
        
        df_A_s2 = pd.DataFrame(rows_A_s2)
        df_A_s2.columns = column_names_A_s2
        df_A_s2.to_csv('{}ds_rssi_M04_v1A_s2.csv'.format(directory), index=False)
        
        df_B_s1 = pd.DataFrame(rows_B_s1)
        df_B_s1.columns = column_names_B_s1
        df_B_s1.to_csv('{}ds_rssi_M04_v1B_s1.csv'.format(directory), index=False)
        
        df_B_s2 = pd.DataFrame(rows_B_s2)
        df_B_s2.columns = column_names_B_s2
        df_B_s2.to_csv('{}ds_rssi_M04_v1B_s2.csv'.format(directory), index=False)
        
        df_C_s1 = pd.DataFrame(rows_C_s1)
        df_C_s1.columns = column_names_C_s1
        df_C_s1.to_csv('{}ds_rssi_M04_v1C_s1.csv'.format(directory), index=False)
        
        df_C_s2 = pd.DataFrame(rows_C_s2)
        df_C_s2.columns = column_names_C_s2
        df_C_s2.to_csv('{}ds_rssi_M04_v1C_s2.csv'.format(directory), index=False)
        
        df_D_s1 = pd.DataFrame(rows_D_s1)
        df_D_s1.columns = column_names_D_s1
        df_D_s1.to_csv('{}ds_rssi_M04_v1D_s1.csv'.format(directory), index=False)
        
        df_D_s2 = pd.DataFrame(rows_D_s2)
        df_D_s2.columns = column_names_D_s2
        df_D_s2.to_csv('{}ds_rssi_M04_v1D_s2.csv'.format(directory), index=False)
        
        df_E_s1 = pd.DataFrame(rows_E_s1)
        df_E_s1.columns = column_names_E_s1
        df_E_s1.to_csv('{}ds_rssi_M04_v1E_s1.csv'.format(directory), index=False)
        
        df_E_s2 = pd.DataFrame(rows_E_s2)
        df_E_s2.columns = column_names_E_s2
        df_E_s2.to_csv('{}ds_rssi_M04_v1E_s2.csv'.format(directory), index=False)
        
        df_F_s1 = pd.DataFrame(rows_F_s1)
        df_F_s1.columns = column_names_F_s1
        df_F_s1.to_csv('{}ds_rssi_M04_v1F_s1.csv'.format(directory), index=False)
        
        df_F_s2 = pd.DataFrame(rows_F_s2)
        df_F_s2.columns = column_names_F_s2
        df_F_s2.to_csv('{}ds_rssi_M04_v1F_s2.csv'.format(directory), index=False)

        # Print progress
        if train_test is 'train' or np.mod(fi, 50) == 0:
            if train_test is 'train': print 
            print ("Finished feature extraction for {}/{}\n".format(train_test, stub_name))




