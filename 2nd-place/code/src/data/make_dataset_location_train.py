# -*- coding: utf-8 -*-
import pandas as pd, numpy as np
import os
from math import floor


##### 
# The goal of this section of code is to create a Pandas dataframe - output_df - containing the location variable of the train set.
#####


# The folder containing the raw data
path_folder = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'data/raw')


#On construit le tableau qui contient, seconde par seconde, la location et les proba de chaque salle
list_directories = os.listdir(path_folder+ '/train')
list_directories.pop(0)
list_directories.sort()

#,Among the 10 individuals, the annotators have not assigned rooms in the same way: some start and finish in the living room, others start and finish in the study, others start and finish in the hall
# We reconstitute manually the list of rooms that every individual visited: since we have the script, we know the order of the rooms 
# Besides, we distinguish two halls: the hall downstairs and the hall upstairs
list_location_datasets = [range(0,3),range(1,3),[0],range(2),[0],range(2),[0],range(2),[0],[1]]
order_1 = ['hall_bas','living','hall_bas','kitchen','hall_bas','living','hall_bas','kitchen','stairs','bed2','hall_haut',
                      'toilet','hall_haut', 'bed1', 'hall_haut', 'bath','stairs','living','hall_bas']
order_2 = ['study','hall_bas','living','hall_bas','kitchen','hall_bas','living','hall_bas','kitchen','stairs','bed2','hall_haut',
                      'toilet','hall_haut', 'bed1', 'hall_haut', 'bath','stairs','living','hall_bas','study']
order_3 = ['living','hall_bas','kitchen','hall_bas','living','hall_bas','kitchen','stairs','bed2','hall_haut',
                      'toilet','hall_haut', 'bed1', 'hall_haut', 'bath','stairs','living']
list_order_datasets = [order_1,order_1,order_1,order_1,order_1,order_1,order_2,order_3,order_2,order_2]


output_df = pd.DataFrame({})
for index,directory in enumerate(list_directories,start=1):

    
    start_df = pd.DataFrame({})
    end_df = pd.DataFrame({})
    location_df = pd.DataFrame({})
    result_df = pd.DataFrame({})
    original_dataset_df = pd.DataFrame({})
    for number in list_location_datasets[index-1]:
            temp_df = pd.read_csv(path_folder+ '/train/'+ directory +'/location_' + str(number) + '.csv')
            start_df['start_'+str(number)] = temp_df['start']
            end_df['end_'+str(number)] = temp_df['end']
    start_df['start'] = start_df.mean(axis = 1)
    start_df.drop(start_df.columns[range(len(list_location_datasets[index-1]))], axis=1, inplace=True)
    end_df['end'] = end_df.mean(axis = 1)
    end_df.drop(end_df.columns[range(len(list_location_datasets[index-1]))], axis=1, inplace=True)
    location_df['location']=list_order_datasets[index-1]
    original_dataset_df['original_dataset'] = [index] * len(start_df)

    result_df = pd.concat([original_dataset_df,start_df, end_df,location_df], axis=1, join_axes=[start_df.index])


    # we create a dataframe that gives the location of the individual at every second
    output_df_locations = pd.DataFrame(columns= ['location'],index = range(len(pd.read_csv(path_folder+ '/train/'+ directory +'/columns.csv')))) 
    output_df_locations.insert(0, 'end', range(1,len(output_df_locations)+1))
    output_df_locations.insert(0, 'start', range(0,len(output_df_locations)))
    output_df_locations.insert(0, 'original_dataset', index)



    for row in result_df.iterrows():
        start = row[1]['start']
        end  = row[1]['end']
        location = row[1]['location']

        interval = range(int(floor(start)),int(floor(end)+1))
        if (len(interval) == 1):
            output_df_locations.set_value(int(floor(start)),'location_'+location, end-start)
            output_df_locations.set_value(int(floor(start)),'location', location)


        elif (len(interval) == 2):
            output_df_locations.set_value(int(floor(start)),'location_'+location, int(floor(start))+1-start)
            output_df_locations.set_value( int(floor(end)), 'location_'+location,end - int(floor(end)))
            output_df_locations.set_value(int(floor(start)),'location', location)


        else: 
            output_df_locations.set_value( int(floor(start)),'location_'+location,int(floor(start))+1-start)
            output_df_locations.set_value( int(floor(end)),'location_'+location, end - int(floor(end)))
            output_df_locations.set_value( int(floor(start)),'location',location)


            inner_interval = range(int(floor(start))+1,int(floor(end)))
            for i in inner_interval:
                output_df_locations.set_value( i,'location_'+location, 1)
                output_df_locations.set_value(i,'location', location)


    output_df = output_df.append(output_df_locations)

    
    
   



# Recipe outputs
output_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'data/interim/location_train.csv')
output_df.to_csv(path_or_buf = output_path , header = True, index = False, encoding='utf-8',)

