# -*- coding: utf-8 -*-
import pandas as pd, numpy as np
import os
from math import floor

#####
# The goal of this section of code is to create the train dataset that we will work with. We append the 10 raw test datasets,
#####

output_df_columns = pd.DataFrame({})

# The folder containing the raw data
path_folder = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'data/raw/')

list_directories = os.listdir(path_folder+ '/train')
list_directories.pop(0)
list_directories.sort()

for index,directory in enumerate(list_directories,start=1):
    df = pd.read_csv(path_folder+ '/train/'+ directory +'/columns.csv')
    df['original_dataset'] = index
    df['start']=range(0,len(df))
    df['end']=range(1,len(df)+1)
    output_df_columns = output_df_columns.append(df)

# Recipe outputs
output_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'data/interim/columns_train.csv')
output_df_columns.to_csv(path_or_buf = output_path , header = True, index = False, encoding='utf-8',)
