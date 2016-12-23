# -*- coding: utf-8 -*-
import pandas as pd, numpy as np
import os
import json
from math import floor
import csv


#####
#The goal of this section of code is to create a Pandas dataframe that gives, for every test dataset, the id of the annotators that labeled the actions
#####

# The folder containing the raw test data
path_folder = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'data/raw/')


list_directories = os.listdir(path_folder+ 'test')
list_directories.sort()
list_directories.pop(0)

output_df_columns = pd.DataFrame(columns=['record_id','annotators'],index = range(len(list_directories)))

for index,directory in enumerate(list_directories,start=11):
    # For every test dataset, we read, as a dictionary, the json file containing the information "annotators id" 
    with open(path_folder+ '/test/'+ directory +'/meta.json') as json_data:
        data = json.load(json_data)
    output_df_columns.loc[index-11]['record_id'] = index
    output_df_columns.loc[index-11]['annotators'] = data['annotators']

# Recipe outputs
output_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'data/interim/annotators_id_test.csv')
output_df_columns.to_csv(path_or_buf = output_path , header = True, index = False, encoding='utf-8')
