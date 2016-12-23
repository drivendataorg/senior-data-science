# -*- coding: utf-8 -*-
import pandas as pd, numpy as np
import os


##### 
# The goal of this section of code is to split the train dataset into multiple small test datasets. The idea is to recreate a structure similar to that of the (multiple and small) test dataset
#####
input_path_1 = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'data/interim/columns_train_with_targets_and_location.csv')
df = pd.read_csv(filepath_or_buffer = input_path_1,encoding='utf-8')
input_path_2 = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'data/interim/info_on_columns_test.csv')
info_on_columns_test_df = pd.read_csv(filepath_or_buffer = input_path_2,encoding='utf-8')


# Make sure that train dataset is ordered by 'original_dataset' and 'start'
df = df.sort(['original_dataset', 'start'])
df = df.reset_index(drop=True)

# Reverse dataframe info_on_columns_test_df
info_on_columns_test_df = info_on_columns_test_df.iloc[::-1]
info_on_columns_test_df = info_on_columns_test_df.reset_index(drop=True)

# Get the distribution of the variable 'length of the test datasets'. Pick up 900 lengths that follow this distribution.
values = info_on_columns_test_df['nb_of_lines'].values.tolist()
probabilities = norm = [float(i)/sum(info_on_columns_test_df['count_of_records'].values.tolist()) for i in info_on_columns_test_df['count_of_records'].values.tolist()]
np.random.seed(28)
length_of_small_datasets = np.random.choice(values, replace = True, size = 900, p=probabilities)


# Split the train dataset: get the first length picked up previously (for example N): create a small train dataset with the first N lines of the train dataset

output_df = pd.DataFrame()
line_index = 0

for i, length in enumerate(length_of_small_datasets,start = 11):
        # Get the first length picked up previously (for example N)
        # We want to create a small train dataset with the first N lines of the train dataset
        # Before doing so, we have to check that the N lines do not exceed the length of the train dataset
        is_line_possible = (line_index +  length <= len(df))
        if (not(is_line_possible)):
            temp_df = df[line_index:len(df)]
            temp_df['final_dataset']=i
            line_index += len(temp_df)
            output_df= output_df.append(temp_df, ignore_index=True)
            break
            
        for j in range(line_index,line_index+length):
        # We also check that the N lines correpond to the same original_dataset
            if ( (df.iloc[j]['original_dataset'] != df.iloc[line_index]['original_dataset']) ): 
                temp_df = df[line_index:j]
                temp_df['final_dataset']=i
                line_index += len(temp_df)
                output_df= output_df.append(temp_df, ignore_index=True)
                is_line_possible = False
                break
        if (not(is_line_possible)):
            continue
        temp_df = df[line_index:line_index+length]
        temp_df['final_dataset']=i
        line_index += length
        output_df= output_df.append(temp_df, ignore_index=True)

        
# Recipe outputs
output_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'data/interim/columns_train_with_targets_and_location_splitted.csv')
output_df.to_csv(path_or_buf = output_path , header = True, index = False, encoding='utf-8')

