from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize

from visualise_data import Sequence
import numpy as np
import os
import pandas as pd
import scipy.stats.stats as st
import warnings
warnings.filterwarnings('ignore')

def energy(arr):
    """
    Energy measure. Sum of the squares divided by the number of values.
    :param arr:
    :return: float
    """
    return np.sum(np.power(arr,2))/len(arr)

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

# We will want to keep track of the feature names for later, so we will collect these in the following list:
column_names = []

# These are the modalities that are available in the dataset, and the .iterate() function returns the data
# in this order
modality_names = ['acceleration', 'rssi', 'pir', 'video_living_room', 'video_kitchen', 'video_hallway']

#acceleration feature
components = 6
feature_pca = ['pca_%d' % i for i in range(components)]
#column_names = feature_pca

feature_functions = [np.mean, np.std, np.min, np.median, np.max, np.var, st.skew, st.kurtosis,
                     st.sem,st.moment,iqr, mad, energy,np.linalg.norm]

feature_names = ['mean', 'std', 'min', 'median', 'max', 'var', 'skew', 'kur',
                 'sem', 'moment','iqr','mad','energy','mag']

feature_names = map(lambda x: "resample_%s" % x, feature_names)

num_ff = len(feature_names)
"""
Iterate over all training directories
"""
for train_test in ('train', 'test',):
    if train_test is 'train':
        print ('Extracting features from training data.\n')
    else:
        print ('\n\n\nExtracting features from testing data.\n')

    for fi, file_id in enumerate(os.listdir('../input/public_data/{}/'.format(train_test))):
        stub_name = str(file_id).zfill(5)
        column_names =[]
        if train_test is 'train' or np.mod(fi, 50) == 0:
            print ("Starting feature extraction for {}/{}".format(train_test, stub_name))


        features = pd.read_csv(os.path.join('../input/public_data/{}/{}'.format(train_test, stub_name), 'columns_v5.csv'))
        features = features.fillna(-9999)
        #print data.shape
        data_normalized = normalize(features.values)
        pca = PCA(n_components=components)
        data_x_projected_pca = pca.fit_transform(features)

        data = Sequence('../input/public_data', '../input/public_data/{}/{}'.format(train_test, stub_name))
        data.load()

        if len(column_names) == 0:
            for lu, modalities in data.iterate():
                for i, (modality, modality_name) in enumerate(zip(modalities, modality_names)):

                    for column_name, column_data in modality.transpose().iterrows():
                        for feature_name in feature_names:
                            column_names.append('{0}_{1}_{2}'.format(modality_name, column_name, feature_name))

                # Break here
                break
        column_names.extend(feature_pca)
        rows = []

        for ri, (lu, modalities) in enumerate(data.iterate()):
            row = []

            for i, modality in enumerate(modalities):
                modality = modality[0:components]
                for name, column_data in modality.transpose().iterrows():
                    if len(column_data) > 3:
                        row.extend(map(lambda ff: ff(column_data), feature_functions))
                    else:
                        row.extend([np.nan] * num_ff)



            rows.append(row)


            # Report progress
            if train_test is 'train':
                if np.mod(ri + 1, 50) == 0:
                    print ("{:5}".format(str(ri + 1))),

            if np.mod(ri + 1, 500) == 0:
                print
        

        
        data = np.hstack((np.array(rows), data_x_projected_pca))
        df = pd.DataFrame(data)
        #print df.head(4)
        assert df.shape[1] == len(column_names)
        df.columns = column_names
        df.to_csv('../input/public_data/{}/{}/columns_v17.csv'.format(train_test, stub_name),
                  index=False)  # if train_test is 'train' or np.mod(fi, 50) == 0:
        if train_test is 'train': print
        print ("Finished feature extraction for {}/{}\n".format(train_test, stub_name))
        #break
