import warnings
import numpy as np
import pandas as pd
import scipy as sp
import scipy.stats.stats as st
from scipy import signal
from scipy.signal import convolve
from spectrum import *

from visualise_data import Sequence
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

#ennerfy fft values
def getEnergy(arr):
    data_fft = sp.fft(arr)
    data_fft_half = data_fft[1:(len(data_fft)/2+1)]
    data_fft_half_abs = np.abs(data_fft_half)
    result = np.sum(data_fft_half_abs**2)
    return result/len(data_fft_half)

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

#calculate the entropy of fft values
def fftEntropy(arr):
    data_fft = sp.fft(arr)
    data_fft_abs = np.abs(data_fft)
    data_fft_abs_sum = np.sum(data_fft_abs)
    data_fft_abs_norm = data_fft_abs/data_fft_abs_sum
    data_fft_abs_norm_log2 = np.log2(data_fft_abs_norm)
    result = - np.sum(data_fft_abs_norm * data_fft_abs_norm_log2)
    result = result/len(data_fft)
    return result

def absCV(arr):
    std = np.std(np.abs(arr))
    mean = np.mean(np.abs(arr))
    return std / mean * 100.0

def widerange(arr):
    return np.max(arr) - np.min(arr)

def fftCoeff(arr):
    return sp.fft(arr)[1:]

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

def absMean(arr):
    return np.mean(np.abs(arr))

def gmean(arr):
	return st.gmean(arr[arr>0])

def pmean(arr):
    return np.mean(arr[arr>0])

def rms(arr):
    return (np.mean(arr**2))**0.5

def p2p(arr):
    return np.amax(arr)-np.amin(arr);

def crestFactor(arr):
    return np.amax(arr)/rms(arr)

def distmax(arr):
    return np.max(arr)-np.mean(arr)

def distmin(arr):
    return np.mean(arr) - np.min(arr)
def domFreqRatio(arr):
    data_fft = sp.fft(arr)
    data_fft_sort = list(np.sort(abs(data_fft[:(len(data_fft)/2+1)])))
    large = data_fft_sort.pop()
    ratio = large / np.sum(data_fft_sort)
    return ratio

def mcr(arr):
    mean = np.mean(arr)
    k = 0
    for i in range(len(arr)-1):
        if (arr[i] - mean) * (arr[i+1] - mean) < 0:
            k += 1
    result = k * 1.0 / (len(arr) - 1)
    return result

#TODO: check
# calculate the second peak of autocorrelation of fft values
def getPitch(arr):
    data_fft = sp.fft(arr)
    result = np.correlate(data_fft, data_fft, 'full')
    result = np.sort(np.abs(result))
    return result[len(result)-2]
#TODO: check
def quartiles(arr):
    q1 = np.percentile(np.abs(arr), 25)
    q2 = np.percentile(np.abs(arr), 50)
    q3 = np.percentile(np.abs(arr), 75)
    return [q1, q2, q3]

#TODO: check
def linRegCoeff(arr):
	x=np.arange(0,len(arr),1)
	A=np.vstack([x, np.ones(len(x))]).T
	a,b=np.linalg.lstsq(A,arr)[0]
	rmse=np.mean(((a*x + b)**2)-(arr**2))
	return a,b,rmse

#TODO: check
def quadRegCoeff(arr):
	x=np.arange(0,len(arr),1)
	A=np.vstack([x**2, x, np.ones(len(x))]).T
	a,b,c=np.linalg.lstsq(A,arr)[0]
	rmse=np.mean(((a*(x**2) + b*x + c)**2)-(arr**2))
	return a,b,c,rmse

#TODO: check
def peaks(arr):
    peaks = signal.find_peaks_cwt(arr,np.arange(1,10,1))
    arr_peaks = arr[peaks]
    pkmean = np.mean(arr_peaks)
    return [pkmean, abs(pkmean-np.mean(arr))]

#TODO: check again
#index of the frequency component with largest magnitude
def maxInds(arr):
    windowSize=max(len(arr),250)
    n=len(arr)
    nWindows=0
    total=np.zeros(windowSize)
    start=0
    while start<(n-windowSize+1):
        total=total+np.fft.fft(arr[start:(start+windowSize)])
        nWindows+=1
        start+=np.floor(windowSize/2)
    meanFourier=total/nWindows
    idx = np.argmax(abs(meanFourier)**2)
    #np.abs(meanFourier[0]) # DC Component
    #np.linalg.norm(meanFourier) # Energy
    #freq = np.fft.fftfreq(len(meanFourier))
    #np.abs(np.sum(meanFourier)) # Coefficients sum
    #freq[idx] #Dominant frequency
    #np.mean(freq) #mean frequency
    return idx

feature_functions = [np.mean, np.std, np.min, np.median, np.max, np.var, st.skew, st.kurtosis,st.sem,st.moment
                     ,iqr,distmin,distmax,crestFactor,p2p,pmean,gmean,absCV,absMean,mad
                     ,domFreqRatio,widerange,entropy,energy,mcr
                     ]
feature_names = ['mean', 'std', 'min', 'median', 'max', 'var', 'skew',' st.kurtosis','sem','moment'
                     ,'iqr','distmin','distmax','crestFactor','p2p','pmean','gmean','absCV','absMean','mad'
                     ,'domFreqRatio','widerange','entropy','energy','mcr'
                     ]
column_names = []
modality_names = ['acceleration', 'rssi', 'pir', 'video_living_room', 'video_kitchen', 'video_hallway']
#acceleration feature
feature_accel = ['acc_ratio','acc_max','acc_min','acc_range', 'acc_sum']
aspect_accel  = ['bodyx','bodyy','bodyz','gravity_acc_x','gravity_acc_y','gravity_acc_z'
                 ,'body_jerkx','body_jerk_y','body_jerk_z','body_mag'
                 #,'fftboby_x','fftbody_y','fft_body_z','fft_body_jerk_x','fft_body_jerk_y','fft_body_jerk_z','fftbody_mag'
                 ]

num_ff_accel = len(feature_accel) + len(feature_names)*len(aspect_accel)

#TODO: jerk signal
def extract_accel(accel):
    if(accel.shape[0]<13): #insufficient
        return [np.nan] * num_ff_accel
    ff_accel = []
    #Signal-Magnitude Area
    '''
    accel['acc'] =  np.sqrt(accel['x']**2 + accel['y']**2 + accel['z']**2)
    arr = accel['acc'].values
    ff_accel.append(arr.max())
    ff_accel.append(arr.min())
    ff_accel.append(np.median(arr))
    ff_accel.append(np.average(arr))
    ff_accel.append(np.sum(arr))'''

    #reference here: https://github.com/danielmurray/adaptiv
    medFilterAccX = signal.medfilt(accel['x'])
    medFilterAccY = signal.medfilt(accel['y'])
    medFilterAccZ = signal.medfilt(accel['z'])

    b, a = signal.butter(3, 10.0/25, btype='low')
    butterAccX = signal.filtfilt(b,a,medFilterAccX)
    butterAccY = signal.filtfilt(b,a,medFilterAccY)
    butterAccZ = signal.filtfilt(b,a,medFilterAccZ)

    b, a = signal.butter(3, 0.3/25, btype='low')
    gravityAccX = signal.filtfilt(b,a,butterAccX)
    gravityAccY = signal.filtfilt(b,a,butterAccY)
    gravityAccZ = signal.filtfilt(b,a,butterAccZ)

    b, a = signal.butter(3, 0.3/25, btype='high')
    bodyAccX = signal.filtfilt(b,a,butterAccX)
    bodyAccY = signal.filtfilt(b,a,butterAccY)
    bodyAccZ = signal.filtfilt(b,a,butterAccZ)

    acc = bodyAccX + bodyAccY + bodyAccZ
    vel = 0

    for i in range(1,len(acc)):
        vel = vel + (acc[i]+acc[i-1])/2.0

    ff_accel.append(float(vel)/len(acc))
    ff_accel.append(max(acc))
    ff_accel.append(min(acc))
    ff_accel.append(max(acc)-min(acc))
    totalAcc = sqrt(np.square(bodyAccX)+np.square(bodyAccY)+np.square(bodyAccZ))
    ff_accel.append(np.mean(totalAcc))


    ff_accel.extend(map(lambda ff: ff(bodyAccX), feature_functions))
    ff_accel.extend(map(lambda ff: ff(bodyAccY), feature_functions))
    ff_accel.extend(map(lambda ff: ff(bodyAccZ), feature_functions))

    ff_accel.extend(map(lambda ff: ff(gravityAccX), feature_functions))
    ff_accel.extend(map(lambda ff: ff(gravityAccY), feature_functions))
    ff_accel.extend(map(lambda ff: ff(gravityAccZ), feature_functions))
    #ff_accel.append(np.sqrt(np.mean(np.square(bodyAccY))))

    #body jerk
    bodyJerkX=np.diff(bodyAccX,n=1)
    bodyJerkY=np.diff(bodyAccY,n=1)
    bodyJerkZ=np.diff(bodyAccZ,n=1)

    ff_accel.extend(map(lambda ff: ff(bodyJerkX), feature_functions))
    ff_accel.extend(map(lambda ff: ff(bodyJerkY), feature_functions))
    ff_accel.extend(map(lambda ff: ff(bodyJerkZ), feature_functions))

    #body jerk mag
    mag = np.linalg.norm([bodyJerkX,bodyJerkY,bodyJerkZ],axis=(1))
    ff_accel.extend(map(lambda ff: ff(mag), feature_functions))

    return ff_accel

accel_feature_functions = [extract_accel]

#TODO: extract features from sensor data from windows that are longer than one second?
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

        if len(column_names) == 0:
            for lu, modalities in data.iterate():
                for i, (modality, modality_name) in enumerate(zip(modalities, modality_names)):
                    if(i==0):
                        column_names.extend(feature_accel)
                        for aspect in aspect_accel:
                            for feature_name in feature_names:
                                column_names.append('{0}_{1}'.format(aspect, feature_name))
                # Break here
                break

        rows = []

        for ri, (lu, modalities) in enumerate(data.iterate()):
            row = []

            for i, modality in enumerate(modalities):
                #accel features
                if(i==0):
                    row.extend(extract_accel(modality))


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

        df = pd.DataFrame(rows)
        df.columns = column_names

        df.to_csv('../input/public_data/{}/{}/columns_v9.csv'.format(train_test, stub_name),
                  index=False)  # if train_test is 'train' or np.mod(fi, 50) == 0:
        if train_test is 'train': print
        print ("Finished feature extraction for {}/{}\n".format(train_test, stub_name))
        #break


