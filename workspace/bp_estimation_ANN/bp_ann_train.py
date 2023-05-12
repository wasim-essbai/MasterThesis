# importing libraries
import numpy as np  # For numerical computation
import pandas as pd  # Data manipulation
import tensorflow as tf
from sklearn.model_selection import train_test_split  # cross validation split
from sklearn.preprocessing import StandardScaler

from bp_ann import create_bp_ann
from keras import optimizers
import warnings

warnings.filterwarnings('ignore')

print(tf.__version__)
print(tf.config.list_physical_devices('GPU'))

data_path = 'F:/Università/Magistrale/Tesi/workspace/dataset'
# data_path = '/content/drive/MyDrive/MasterThesis/workspace/dataset'

# Loading the dataset
dataset = pd.read_csv(f'{data_path}/dataset_part{1}.csv')
print(f'dataset Data type: {type(dataset)}')
print(f'dataset shape/dimensions: {dataset.shape}')

X = dataset.iloc[0:, 2:].to_numpy()
y = dataset.iloc[0:, 0:2].to_numpy()

# creating train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, shuffle=False)

input_dim = X_train.shape[1]
print('Input size', input_dim)

activation = 'relu'
num_classes = y_train.shape[1]
bp_ann = create_bp_ann(input_dim=input_dim, activation=activation, num_class=num_classes)
bp_ann.compile(loss='MeanAbsoluteError',
               optimizer=optimizers.Adam(lr=0.001),
               metrics=['MeanAbsoluteError', 'MeanAbsolutePercentageError'])
bp_ann.summary()

# Training the model
history = bp_ann.fit(X_train,
                     y_train,
                     epochs=5,
                     batch_size=128,
                     verbose=1)

print("Training done!")
bp_ann.save('./workspace/bp_estimation_ANN/model/bp_ann_model')
np.save('/content/drive/MyDrive/MasterThesis/workspace/ann_dataset/x_train', X_train)
np.save('/content/drive/MyDrive/MasterThesis/workspace/ann_dataset/y_train', y_train)
np.save('/content/drive/MyDrive/MasterThesis/workspace/ann_dataset/x_test', X_test)
np.save('/content/drive/MyDrive/MasterThesis/workspace/ann_dataset/y_test', y_test)
