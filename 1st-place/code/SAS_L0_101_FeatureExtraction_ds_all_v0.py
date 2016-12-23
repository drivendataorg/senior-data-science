# -*- coding: utf-8 -*-

# For number crunching
import numpy as np
import pandas as pd

# Misc
import os

from func_visualise_data import Sequence

import warnings
warnings.filterwarnings('ignore')

"""
For every data modality, we will extract some very simple features: the mean, min, max, median, and standard 
deviation of one second windows. We will put function pointers in a list called 'feature_functions' so that we 
can easily call these on our data for later
"""
feature_functions = [np.mean, np.std, np.min, np.median, np.max]
feature_names = ['mean', 'std', 'min', 'median', 'max']

# We will keep the number of extracted feature functions as a parameter 
num_ff = len(feature_functions)

# We will want to keep track of the feature names for later, so we will collect these in the following list: 
column_names = []

# These are the modalities that are available in the dataset, and the .iterate() function returns the data 
# in this order
modality_names = ['acceleration', 'rssi', 'pir', 'video_living_room', 'video_kitchen', 'video_hallway']

"""
Iterate over all training directories
"""
for train_test in ('train', 'test', ): 
 
     if train_test is 'train': 
         print ('Extracting features from training data.\n')
     else: 
         print ('\n\n\nExtracting features from testing data.\n')
        
     for fi, file_id in enumerate(os.listdir('../public_data/{}/'.format(train_test))):
        stub_name = str(file_id).zfill(5)

        if train_test == 'train' or np.mod(fi, 50) == 0:
             print ("Starting feature extraction for {}/{}".format(train_test, stub_name))

        # Use the sequence loader to load the data from the directory. 
        data = Sequence('../public_data', '../public_data/{}/{}'.format(train_test, stub_name))
        data.load()

        """
        Populate the column_name list here. This needs to only be done on the first iteration
        because the column names will be the same between datasets. 
        """
        if len(column_names) == 0:
            for lu, modalities in data.iterate():
                for modality, modality_name in zip(modalities, modality_names):
                    for column_name, column_data in modality.transpose().iterrows():
                        for feature_name in feature_names:
                            column_names.append('{0}_{1}_{2}'.format(modality_name, column_name, feature_name))

                # Break here 
                break 

        """
        Here, we will extract some features from the data. We will use the Sequence.iterate function. 

        This function partitions the data into one-second dataframes for the full data sequence. The first element 
        (which we call lu) represents the lower and upper times of the sliding window, and the second is a list of
        dataframes, one for each modality. The dataframes may be empty (due to missing data), and feature extraction 
        has to be able to cope with this! 

        The list rows will store the features extracted for this dataset
        """
        rows = []

        for ri, (lu, modalities) in enumerate(data.iterate()):
            row = []

            """
            Iterate over the sensing modalities. The order is given in modality_names. 

            Note: If you want to treat each modality differently, you can do the following: 

            for ri, (lu, (accel, rssi, pir, vid_lr, vid_k, vid_h)) in enumerate(data.iterate()):
                row.extend(extract_accel(accel))
                row.extend(extract_rssi(rssi))
                row.extend(extract_pir(pir))
                row.extend(extract_video(vid_lr, vid_k, vid_h))

            where the extract_accel/extract_rssi/extract_pir/extract_video functions are designed by you. 
            In the case here, we extract the same features from each modality, but optimal performance will 
            probably be achieved by considering different features for each modality. 
            """

            for modality in modalities:
                """
                The accelerometer dataframe, for example, has three columns: x, y, and z. We want to extract features 
                from all of these, and so we iterate over the columns here. 
                """
                for name, column_data in modality.transpose().iterrows():
                    if len(column_data) > 3:
                        """
                        Extract the features stored in feature_functions on the column data if there is sufficient 
                        data in the dataframe. 
                        """
                        row.extend(map(lambda ff: ff(column_data), feature_functions))

                    else:
                        """
                        If no data is available, put nan placeholders to keep the column widths consistent
                        """
                        row.extend([np.nan] * num_ff)

            # Do a quick sanity check to ensure that the feature names and number of extracted features match
            assert len(row) == len(column_names)

            # Append the row to the full set of features
            rows.append(row)

            # Report progress 
            if train_test is 'train':
                if np.mod(ri + 1, 50) == 0:
                    print ("{:5}".format(str(ri + 1))),

                if np.mod(ri + 1, 500) == 0:
                    print

        """
        At this stage we have extracted a bunch of simple features from the data. In real implementation, 
        it would be advisable to look at more interesting features, eg

          * acceleration: link
          * environmental: link
          * video: link

        We will save these features to a new file called 'columns.csv' for use later. This file will be located 
        in the name of the training sequence. 
        """
        df = pd.DataFrame(rows)
        df.columns = column_names
        directory = '../preprocessed_data/{}/{}/'.format(train_test, stub_name)
        if not os.path.exists(directory):
            os.makedirs(directory)
        df.iloc[:,0:15].to_csv('{}ds_accel_v0.csv'.format(directory), index=False)
        df.iloc[:,15:35].to_csv('{}ds_rssi_v0.csv'.format(directory), index=False)
        df.iloc[:,35:80].to_csv('{}ds_pir_v0.csv'.format(directory), index=False)
        df.iloc[:,80:305].to_csv('{}ds_video_v0.csv'.format(directory), index=False)

        if train_test is 'train' or np.mod(fi, 50) == 0:
            if train_test is 'train': print 
            print ("Finished feature extraction for {}/{}\n".format(train_test, stub_name))