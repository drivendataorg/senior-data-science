# -*- coding: utf-8 -*-
"""
Created on Thu Jul  7 06:04:50 2016

"""

# For number crunching
import numpy as np
import pandas as pd

# Misc
import os

import warnings
warnings.filterwarnings('ignore')

# load custom libraries
import func_visualise_data as fvd
#reload(fvd)

"""
Iterate over all training directories
"""
for train_test in ('train', ):
     if train_test is 'train':
         print ('Extracting features from training data.\n')
     else:
         print ('\n\n\nExtracting features from testing data.\n')

     for fi, file_id in enumerate(sorted(os.listdir('../public_data/{}/'.format(train_test)))):
        stub_name = str(file_id).zfill(5)

        if train_test == 'train' or np.mod(fi, 50) == 0:
             print ("Starting feature extraction for {}/{}".format(train_test, stub_name))

        # Use the sequence loader to load the data from the directory.
        data = fvd.Sequence('../public_data', '../public_data/{}/{}'.format(train_test, stub_name))
        data.load()

        # Load location
        start_end = []
        locations = []
        #for ri, (lu, (pir, loc)) in enumerate(data.iterate_location()):
        for ri, (lu, (loc))  in enumerate(data.iterate_location()):
            locations.append(loc)
            start_end.append(lu)

        locations = np.vstack(locations)
        start_end = np.vstack(start_end)

        # normalize class probabilities
        locations = np.apply_along_axis(lambda x:x/sum(x), 1, locations)

        # save data
        df = pd.DataFrame(np.concatenate((start_end, locations), axis=1))
        df.columns = ['start', 'end'] + data.location_targets
        directory = '../preprocessed_data/{}/{}/'.format(train_test, stub_name)
        if not os.path.exists(directory):
            os.makedirs(directory)
        df.to_csv('{}targets_locations.csv'.format(directory), index=False)
