from visualise_data import Sequence
import numpy as np
import os

import pandas as pd
import warnings
import scipy.stats.stats as st
from scipy import spatial
from scipy.signal import convolve
import scipy.spatial.distance as distance
from spectrum import *
warnings.filterwarnings('ignore')

def peak(arr):
    """
    Reference: http://stackoverflow.com/questions/24656367/find-peaks-location-in-a-spectrum-numpy
    :param arr:
    :return:
    """
    #Obtaining derivative
    kernel = [1, 0, -1]
    dY = convolve(arr, kernel, 'valid')

    #Checking for sign-flipping
    S = np.sign(dY)
    ddS = convolve(S, kernel, 'valid')

    #These candidates are basically all negative slope positions
    #Add one since using 'valid' shrinks the arrays
    candidates = np.where(dY < 0)[0] + (len(kernel) - 1)

    #Here they are filtered on actually being the final such position in a run of
    #negative slopes
    peaks = sorted(set(candidates).intersection(np.where(ddS == 2)[0] + 1))
    return len(peaks)

def zero_crossing(arr):
    """
    refereence: http://stackoverflow.com/questions/3843017/efficiently-detect-sign-changes-in-python
    :param arr:
    :return:
    """
    return len(np.where(numpy.diff(np.sign(arr)))[0])

def energy(arr):
    """
    Energy measure. Sum of the squares divided by the number of values.
    :param arr:
    :return: float
    """
    return np.sum(np.power(arr,2))/len(arr)

def entropy(arr):
    """
    Reference: https://www.hdm-stuttgart.de/~maucher/Python/MMCodecs/html/basicFunctions.html
    :param arr:
    :return:
    """
    lensig = len(arr)
    symset = list(set(arr))
    propab = [np.size(arr[arr==i])/(1.0*lensig) for i in symset]
    ent = np.sum([p*np.log2(1.0/p) for p in propab])
    return ent

def mad(arr):
    """
        Median Absolute Deviation: https://en.wikipedia.org/wiki/Median_absolute_deviation
        http://stackoverflow.com/questions/8930370/where-can-i-find-mad-mean-absolute-deviation-in-scipy
    :param arr:
    :return: float
    """
    arr = np.ma.array(arr).compressed() # should be faster to not use masked arrays.
    med = np.median(arr)
    return np.median(np.abs(arr - med))

def iqr(arr):
    """
    Interquartile Range: http://stackoverflow.com/questions/23228244/how-do-you-find-the-iqr-in-numpy
    :param arr:
    :return:
    """
    q75, q25 = np.percentile(arr, [75 ,25])
    return q75 - q25

"""
For every data modality, we will extract some very simple features: the mean, min, max, median, and standard 
deviation of one second windows. We will put function pointers in a list called 'feature_functions' so that we 
can easily call these on our data for later
"""
feature_functions = [np.mean, np.std, np.min, np.median, np.max, np.var, st.skew, st.kurtosis,
                     st.sem,st.moment,iqr, mad, energy,np.linalg.norm]
feature_names = ['mean', 'std', 'min', 'median', 'max', 'var', 'skew', 'kur',
                 'sem', 'moment','iqr','mad','energy','mag']
                 
feature_names = map(lambda x: "diff_%s" % x, feature_names)

# We will keep the number of extracted feature functions as a parameter 
num_ff = len(feature_functions)

# We will want to keep track of the feature names for later, so we will collect these in the following list: 
column_names = []

# These are the modalities that are available in the dataset, and the .iterate() function returns the data 
# in this order
modality_names = ['acceleration', 'rssi', 'pir', 'video_living_room', 'video_kitchen', 'video_hallway']

#TODO: extract features from sensor data from windows that are longer than one second?
"""
Iterate over all training directories
"""
for train_test in ('train','test'):
    if train_test is 'train':
        print ('Extracting features from training data.\n')
    else:
        print ('\n\n\nExtracting features from testing data.\n')

    for fi, file_id in enumerate(os.listdir('../input/public_data/{}/'.format(train_test))):
        stub_name = str(file_id).zfill(5)

        if train_test is 'train' or np.mod(fi, 50) == 0:
            print ("Starting feature extraction for {}/{}".format(train_test, stub_name))

        # Use the sequence loader to load the data from the directory.
        data = Sequence('../input/public_data', '../input/public_data/{}/{}'.format(train_test, stub_name))
        data.load()

        """
        Populate the column_name list here. This needs to only be done on the first iteration
        because the column names will be the same between datasets.
        """
        if len(column_names) == 0:
            for lu, modalities in data.iterate():
                for i, (modality, modality_name) in enumerate(zip(modalities, modality_names)):
                    
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

            for i, modality in enumerate(modalities):
                """
                The accelerometer dataframe, for example, has three columns: x, y, and z. We want to extract features
                from all of these, and so we iterate over the columns here.
                """
                
                modality = modality.diff()
                
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
                
            #break
            # Do a quick sanity check to ensure that the feature names and number of extracted features match
            #print len(row), len(column_names)
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
        #temp is v5
        df.to_csv('../input/public_data/{}/{}/columns_v8.csv'.format(train_test, stub_name),
                  index=False)  # if train_test is 'train' or np.mod(fi, 50) == 0:
        if train_test is 'train': print
        print ("Finished feature extraction for {}/{}\n".format(train_test, stub_name))
        #break

'''
--x,y,z--
#entropy
--xyz--
#Fast Fourier Transform (FFT)
'''
