# -*- coding: utf-8 -*-
import pandas as pd, numpy as np
import os
from math import floor

##### 
# The goal of this section of code is to create a Pandas dataframe - output_df_targets - containing the target variable of the train tests.
#####

# The folder containing the raw data
path_folder = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'data/raw/')

list_directories = os.listdir(path_folder+ '/train')
list_directories.pop(0)
list_directories.sort()

output_df_targets = pd.DataFrame({})

for index,directory in enumerate(list_directories,start=1):
    df = pd.read_csv(path_folder+ '/train/'+ directory + '/targets.csv')
    df['original_dataset'] = index
    output_df_targets = output_df_targets.append(df)

# The raw target consists of 20 columns: it is probabilistic.
# We convert the target into a categorical variable, i.e. one column, by keeping the most probable value
targets_df_prepared = output_df_targets.drop(['start','end','original_dataset'], 1)
result_series = targets_df_prepared.idxmax(axis=1, skipna=False)
result_dataframe = result_series.to_frame()
result_dataframe.rename(columns={result_dataframe.columns.values[0]:"target"}, inplace=True)

output_df_targets = pd.concat([output_df_targets, result_dataframe], axis=1)

# Recipe outputs
output_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'data/interim/targets_train.csv')
output_df_targets.to_csv(path_or_buf = output_path , header = True, index = False, encoding='utf-8',)
