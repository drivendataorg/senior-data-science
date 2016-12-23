# -*- coding: utf-8 -*-
import pandas as pd, numpy as np
import os
from math import floor


#####
# The goal of this section of code is to create the test dataset that we will work on. We append the 872 raw test datasets.
#####

output_df_columns = pd.DataFrame({})

# The folder containing the raw data
path_folder = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'data/raw/')

list_directories = os.listdir(path_folder+ '/test')
list_directories.sort()
list_directories.pop(0)

for index,directory in enumerate(list_directories,start=11):
    df = pd.read_csv(path_folder+ '/test/'+ directory +'/columns.csv')
    df['record_id'] = index
    df['start']=range(0,len(df))
    df['end']=range(1,len(df)+1)
    df.drop(df.index[-1:], inplace=True)
    output_df_columns = output_df_columns.append(df)

# Recipe outputs
output_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'data/interim/columns_test.csv')
output_df_columns.to_csv(path_or_buf = output_path , header = True, index = False, encoding='utf-8',)
