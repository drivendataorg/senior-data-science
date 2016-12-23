# -*- coding: utf-8 -*-
import pandas as pd, numpy as np
from collections import defaultdict
import os

#####
# The goal of this section of code is to create a Pandas Dataframe - length_df - that contains the following information:
# - the different lengths of the test datasets
# - for each length N, how many test datasets have length N
#####

# Recipe inputs
folder_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'data/raw/')

list_directories = os.listdir(folder_path+ '/test')
list_directories.sort()
list_directories.pop(0)

my_dict = defaultdict(int)
for index,directory in enumerate(list_directories,start=11):
    df = pd.read_csv(folder_path+ '/test/'+ directory +'/columns.csv')
    df.drop(df.index[-1:], inplace=True)
    my_dict[len(df)]+=1
    
length_df = pd.Series(my_dict, name='count_of_records')
length_df.index.name = 'nb_of_lines'
length_df = length_df.reset_index()

# Recipe outputs
output_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'data/interim/info_on_columns_test.csv')
length_df.to_csv(path_or_buf = output_path , header = True, index = False, encoding='utf-8',)
