# importing libraries
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from keras import backend as K
from bp_ann import create_bp_ann
from keras import optimizers
import warnings
import pickle

warnings.filterwarnings('ignore')

print(tf.__version__)
print(tf.config.list_physical_devices('GPU'))
device_name = tf.test.gpu_device_name()

loss_name = 'MAE'

# data_path = 'C:/Users/Wasim/Documents/Universita/Magistrale/Tesi/workspace/data_split'
data_path = './workspace/data_split'

# Loading the dataset
X_train = pd.read_csv(f'{data_path}/X_train.csv', header=None).to_numpy()
y_train = pd.read_csv(f'{data_path}/y_train.csv', header=None).to_numpy()
X_val = pd.read_csv(f'{data_path}/X_val.csv', header=None).to_numpy()
y_val = pd.read_csv(f'{data_path}/y_val.csv', header=None).to_numpy()
X_test = pd.read_csv(f'{data_path}/X_test.csv', header=None).to_numpy()
y_test = pd.read_csv(f'{data_path}/y_test.csv', header=None).to_numpy()

print(f'Train dataset : {y_train.shape[0]}')
print(f'Validation dataset : {y_val.shape[0]}')
print(f'Test dataset : {y_test.shape[0]}')

input_dim = X_train.shape[1]
print('Input size', input_dim)

activation = 'relu'
num_classes = y_train.shape[1]
bp_ann = create_bp_ann(input_dim=input_dim, activation=activation, num_class=num_classes)


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


print('Loss function employed: ', loss_name)
if loss_name == 'MAE':
    bp_ann.compile(loss='MeanAbsoluteError',
                   optimizer=optimizers.RMSprop(learning_rate=0.007),
                   metrics=[
                            'MeanAbsoluteError',
                            'MeanAbsolutePercentageError',
                            MAE_SBP, MAE_DBP, MAE_MBP])
else:
    bp_ann.compile(loss='MeanAbsolutePercentageError',
                   optimizer=optimizers.RMSprop(learning_rate=0.007),
                   metrics=[
                       'MeanAbsoluteError',
                       'MeanAbsolutePercentageError',
                       MAE_SBP, MAE_DBP, MAE_MBP])
bp_ann.summary()

# Training the model
if device_name == '/device:GPU:0':
    with tf.device('/device:GPU:0'):
        print('Training using GPU')
        history = bp_ann.fit(X_train,
                             y_train,
                             epochs=40,
                             shuffle=True,
                             batch_size=32,
                             verbose=2)
else:
    print('Training using CPU')
    history = bp_ann.fit(X_train,
                         y_train,
                         epochs=40,
                         shuffle=True,
                         batch_size=32,
                         verbose=2)
print("Training done!")

bp_ann.save('./workspace/bp_estimation_ANN/model/bp_ann_model_' + loss_name)
with open('./workspace/bp_estimation_ANN/trainHistoryDict_' + loss_name, 'wb') as file_pi:
    pickle.dump(history.history, file_pi)

print("Evaluate on validation data")
results = bp_ann.evaluate(X_val, y_val, batch_size=32)
print(results)
print("Evaluate on test data")
test_results = bp_ann.evaluate(X_test, y_test, batch_size=32)
print(test_results)
