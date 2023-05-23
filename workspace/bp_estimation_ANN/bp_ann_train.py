# importing libraries
import numpy as np  # For numerical computation
import pandas as pd  # Data manipulation
import tensorflow as tf
from sklearn.model_selection import train_test_split  # cross validation split
from sklearn.preprocessing import StandardScaler
from keras import backend as K
from bp_ann import create_bp_ann
from keras import optimizers
import warnings

warnings.filterwarnings('ignore')

print(tf.__version__)
print(tf.config.list_physical_devices('GPU'))

# data_path = 'F:/Università/Magistrale/Tesi/workspace/dataset'
data_path = '/content/drive/MyDrive/MasterThesis/workspace/dataset'

# Loading the dataset
dataset1 = pd.read_csv(f'{data_path}/dataset_part{1}.csv')
dataset2 = pd.read_csv(f'{data_path}/dataset_part{2}.csv')
dataset3 = pd.read_csv(f'{data_path}/dataset_part{3}.csv')
dataset4 = pd.read_csv(f'{data_path}/dataset_part{4}.csv')
dataset5 = pd.read_csv(f'{data_path}/dataset_part{5}.csv')

dataset = pd.concat([dataset1, dataset2, dataset3, dataset4, dataset5])
print(f'dataset Data type: {type(dataset)}')
print(f'dataset shape/dimensions: {dataset.shape}')

features_to_exclude = ['st10','st25','st33','st50','st66','st75']
dataset = dataset.loc[:, ~dataset.columns.isin(features_to_exclude)]

X = dataset.iloc[0:, 3:].to_numpy()
y = dataset.iloc[0:, 1:3].to_numpy()

# creating train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, shuffle=False)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, shuffle=False)

input_dim = X_train.shape[1]
print('Input size', input_dim)

activation = 'relu'
num_classes = y_train.shape[1]
bp_ann = create_bp_ann(input_dim=input_dim, activation=activation, num_class=num_classes)


def MAE_SBP(y_true, y_pred):
    return K.mean(K.abs(y_pred[:, 0] - y_true[:, 0]))

def STD_SBP(y_true, y_pred):
    return K.std(K.abs(y_pred[:, 0] - y_true[:, 0]))

def MAE_DBP(y_true, y_pred):
    return K.mean(K.abs(y_pred[:, 1] - y_true[:, 1]))

def STD_DBP(y_true, y_pred):
    return K.std(K.abs(y_pred[:, 1] - y_true[:, 1]))

bp_ann.compile(loss='MeanAbsoluteError',
               optimizer=optimizers.Adam(lr=0.001),
               metrics=['MeanAbsolutePercentageError', 
               MAE_SBP, STD_SBP, MAE_DBP, STD_DBP])
               
bp_ann.summary()

# Training the model
history = bp_ann.fit(X_train,
                     y_train,
                     epochs=100,
                     batch_size=64,
                     verbose=2)

print("Training done!")
bp_ann.save('./workspace/bp_estimation_ANN/model/bp_ann_model')
np.save('/content/drive/MyDrive/MasterThesis/workspace/ann_dataset/x_train', X_train)
np.save('/content/drive/MyDrive/MasterThesis/workspace/ann_dataset/y_train', y_train)
np.save('/content/drive/MyDrive/MasterThesis/workspace/ann_dataset/x_test', X_test)
np.save('/content/drive/MyDrive/MasterThesis/workspace/ann_dataset/y_test', y_test)

print("Evaluate on validation data")
results = bp_ann.evaluate(X_val, y_val, batch_size=64)
print(results)