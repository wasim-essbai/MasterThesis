# importing libraries
import numpy as np
import scipy.io
import pandas as pd
import tensorflow as tf
import warnings
from keras import backend as K
from sklearn.model_selection import train_test_split
from bp_bnn import create_bp_bnn
from keras import optimizers

warnings.filterwarnings('ignore')

print(tf.__version__)
print(tf.config.list_physical_devices('GPU'))
device_name = tf.test.gpu_device_name()

# data_path = 'C:/Users/Wasim/Documents/Universita/Magistrale/Tesi/workspace/ppg_feature_extraction'
data_path = './workspace/ppg_feature_extraction/dataset_extracted'

# Loading the dataset
dataset1 = pd.read_csv(f'{data_path}/dataset_part{1}.csv')
dataset2 = pd.read_csv(f'{data_path}/dataset_part{2}_rest.csv')
dataset3 = pd.read_csv(f'{data_path}/dataset_part{3}.csv')
dataset4 = pd.read_csv(f'{data_path}/dataset_part{4}.csv')

dataset = pd.concat([dataset2])
print(f'dataset Data type: {type(dataset)}')
print(f'dataset shape/dimensions: {dataset.shape}')

features_to_exclude = []
features_to_exclude = ['st10', 'st25', 'st33', 'st50', 'st66', 'st75']
# features_to_exclude.extend(['st10_p_dt10'])
# features_to_exclude.extend(['st25_p_dt25'])
# features_to_exclude.extend(['st33_p_dt33'])
# features_to_exclude.extend(['st50_p_dt50'])
# features_to_exclude.extend(['st66_p_dt66'])
# features_to_exclude.extend(['st75_p_dt75'])
dataset = dataset.loc[:, ~dataset.columns.isin(features_to_exclude)]

X = dataset.iloc[0:, 4:].to_numpy()
y = dataset.iloc[0:, 0:4].to_numpy()

# creating train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, shuffle=False)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, shuffle=True)

y_train_ids = y_train[0:, 0]
y_val_ids = y_val[0:, 0]
y_test_ids = y_test[0:, 0]

y_train = y_train[0:, 2:]
y_val = y_val[0:, 2:]
y_test = y_test[0:, 2:]

input_dim = X_train.shape[1]
print('Input size', input_dim)

activation = 'relu'
num_classes = y_train.shape[1]
bp_bnn = create_bp_bnn(input_dim=input_dim, activation=activation, train_size=X_train.shape[0], num_class=num_classes)


# loss function definition
def negative_loglikelihood(targets, estimated_distribution):
    return -estimated_distribution.log_prob(targets)


# Metrics to evaluate
def MAE_SBP(y_true, y_pred):
    return K.mean(K.abs(y_pred[:, 0] - y_true[:, 0]))


def STD_SBP(y_true, y_pred):
    return K.std(K.abs(y_pred[:, 0] - y_true[:, 0]))


def MAE_DBP(y_true, y_pred):
    return K.mean(K.abs(y_pred[:, 1] - y_true[:, 1]))


def STD_DBP(y_true, y_pred):
    return K.std(K.abs(y_pred[:, 1] - y_true[:, 1]))


def MAE_MBP(y_true, y_pred):
    mbp_pred = y_pred[:, 0] / 3 + y_pred[:, 0] * 2 / 3
    mbp_true = y_true[:, 0] / 3 + y_true[:, 0] * 2 / 3
    return K.mean(K.abs(mbp_true - mbp_pred))


# bp_bnn.compile(loss='MeanAbsoluteError',
#               optimizer=optimizers.Adam(lr=0.004),
#               metrics=['MeanAbsolutePercentageError',
#                        MAE_SBP, MAE_DBP, MAE_MBP])
bp_bnn.compile(loss=negative_loglikelihood,
               optimizer=optimizers.RMSprop(learning_rate=0.0005, 
               clipnorm=1.0, momentum=0.1),
               metrics=['MeanAbsoluteError', 
                        'MeanAbsolutePercentageError',
                        MAE_SBP, MAE_DBP, MAE_MBP])

bp_bnn.summary()

# Training the model
if device_name == '/device:GPU:0':
    with tf.device('/device:GPU:0'):
        print('Training using GPU')
        history = bp_bnn.fit(X_train,
                             y_train,
                             epochs=400,
                             batch_size=32,
                             verbose=2)
else:
    print('Training using CPU')
    history = bp_bnn.fit(X_train,
                         y_train,
                         epochs=400,
                         batch_size=32,
                         verbose=2)

print("Training done!")
prediction_distribution = bp_bnn(X_test)
print(prediction_distribution)

bp_bnn.save('./workspace/bp_estimation_BNN/model/bp_bnn_model')
np.savetxt('./workspace/bp_estimation_BNN/data_split/y_train_ids.csv', y_train_ids, delimiter=',')
np.savetxt('./workspace/bp_estimation_BNN/data_split/y_val_ids.csv', y_val_ids, delimiter=',')
np.savetxt('./workspace/bp_estimation_BNN/data_split/y_test_ids.csv', y_test_ids, delimiter=',')

print("Evaluate on validation data")
results = bp_bnn.evaluate(X_val, y_val, batch_size=32)
print(results)
