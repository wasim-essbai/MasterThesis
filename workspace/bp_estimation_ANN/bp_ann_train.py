# importing libraries
import numpy as np  # For numerical computation
import pandas as pd  # Data manipulation
import seaborn as sns  # plotting
import scipy.io  # reading matlab files in python
import tensorflow as tf
from scipy import signal  # signal processing
from scipy.fftpack import fft, dct  # signal processing
from sklearn.linear_model import LinearRegression  # linear regression model
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, train_test_split  # cross validation split
from sklearn.metrics import mean_squared_error
from bp_ann import create_bp_ann
from keras import optimizers
import warnings

warnings.filterwarnings('ignore')

from matplotlib import pyplot as plt  # For plotting graphs(Visualization)

import os  # system-wide functions

print(tf.__version__)
print(tf.config.list_physical_devices('GPU'))

#data_path = 'F:/Università/Magistrale/Tesi/workspace/dataset'
data_path = '/content/drive/MyDrive/MasterThesis/workspace/dataset'

# Loading a sample .mat file to understand the data dimensions
test_sample = scipy.io.loadmat(f'{data_path}/part_{1}.mat')['p']
print(f'test_sample Data type: {type(test_sample)}')
print(f'test_sample shape/dimensions: {test_sample.shape}')

## Try printing out the entire array to see its data.
print(f"Total Samples: {len(test_sample[0])}")
print(f"Number of readings in each sample(column): {len(test_sample[0][0])}")
print(f"Number of samples in each reading(ECG): {len(test_sample[0][0][2])}")

temp_mat = test_sample[0, 999]
temp_length = temp_mat.shape[1]
sample_size = 125

print(temp_length)
print(int(temp_length / sample_size))

sample_size = 125
ppg = []
for i in range(1000):
    temp_mat = test_sample[0, i]
    temp_length = temp_mat.shape[1]
    for j in range(int(temp_length / sample_size)):
        temp_ppg = temp_mat[0, j * sample_size:(j + 1) * sample_size]
        ppg.append(temp_ppg)

ecg = []
bp = []
sbp = []  # Systolic Blood Pressure
dbp = []  # Diastolic Blood Pressue
size = 125  # sample size

for i in range(1000):
    temp_mat = test_sample[0, i]
    temp_length = temp_mat.shape[1]
    for j in range(int(temp_length / sample_size)):
        temp_ecg = temp_mat[2, j * size:(j + 1) * size]
        temp_bp = temp_mat[1, j * size:(j + 1) * size]

        max_value = max(temp_bp)
        min_value = min(temp_bp)

        sbp.append(max_value)
        dbp.append(min_value)
        ecg.append(temp_ecg)
        bp.append(temp_bp)

# Reshaping the ecg, ppg and bp signal data into column vectors
ppg, ecg, bp = np.array(ppg).reshape(-1, 1), np.array(ecg).reshape(-1, 1), np.array(bp).reshape(-1, 1)
sbp, dbp = np.array(sbp).reshape(-1, 1), np.array(dbp).reshape(-1, 1)
print(f'PPG_shape: {ppg.shape}\n ECG_shape: {ecg.shape}\n BP_shape: {bp.shape}')
print(f'Systolic-BP_shape: {sbp.shape},\n Diastolic-BP_shape: {dbp.shape}')

##plotting sample ppg, ecg and bp signals
##using a sample size of 125
fig, ax = plt.subplots(3, 1, figsize=(9, 12), sharex=True)

ax[0].set_title('PPG graph', fontsize=16)
ax[0].set_ylabel('Signal Value')
ax[0].plot(ppg[:125])

ax[1].set_title('ECG graph', fontsize=16)
ax[1].set_ylabel('Signal Value')
ax[1].plot(ecg[:125])

ax[2].set_title('Blood Pressure (BP) graph', fontsize=16)
ax[2].set_ylabel('Signal Value')
ax[2].set_xlabel('Sample size')
ax[2].plot(bp[:125])

plt.show()

# creating train and test sets
X_train, X_test, y_train, y_test = train_test_split(ppg, bp, test_size=0.30)

input_dim = X_train.shape[1]
print('Input size', input_dim)
activation = 'relu'
num_classes = 1
bp_ann = create_bp_ann(input_dim=input_dim, activation=activation, num_class=num_classes)
bp_ann.compile(loss='Huber',
               optimizer=optimizers.Adam(lr=0.001),
               metrics=['MeanAbsoluteError'])
bp_ann.summary()

# Training the model
history = bp_ann.fit(X_train,  # using the first 1million rows for speed.
                     y_train.squeeze(),
                     epochs=5,
                     batch_size=1024,
                     verbose=1)

print("Training done!")
bp_ann.save('./MasterThesis/workspace/bp_estimation_ANN/model/bp_ann_model')
np.save('/content/drive/MyDrive/MasterThesis/workspace/ann_dataset/x_train',X_train)
np.save('/content/drive/MyDrive/MasterThesis/workspace/ann_dataset/y_train',y_train)
np.save('/content/drive/MyDrive/MasterThesis/workspace/ann_dataset/x_test',X_test)
np.save('/content/drive/MyDrive/MasterThesis/workspace/ann_dataset/y_test',y_test)