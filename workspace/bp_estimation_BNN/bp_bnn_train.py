# importing libraries
import numpy as np
import pandas as pd
import tensorflow as tf
import warnings
from keras import backend as K
from bp_bnn import create_bp_bnn
from keras import optimizers

warnings.filterwarnings('ignore')

print(tf.__version__)
print(tf.config.list_physical_devices('GPU'))
device_name = tf.test.gpu_device_name()

loss_name = 'MAPE'

data_path = 'C:/Users/Wasim/Documents/Universita/Magistrale/Tesi/workspace/data_split'
# data_path = './workspace/data_split'

# Loading the dataset
X_train = pd.read_csv(f'{data_path}/X_train.csv', header=None)
y_train = pd.read_csv(f'{data_path}/y_train.csv', header=None)
X_val = pd.read_csv(f'{data_path}/X_val.csv', header=None)
y_val = pd.read_csv(f'{data_path}/y_val.csv', header=None)
X_test = pd.read_csv(f'{data_path}/X_test.csv', header=None)
y_test = pd.read_csv(f'{data_path}/y_test.csv', header=None)

print(f'Train dataset : {y_train.shape[0]}')
print(f'Validation dataset : {y_val.shape[0]}')
print(f'Test dataset : {y_test.shape[0]}')

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


print('Loss function employed: ', loss_name)
if loss_name == 'MAE':
    bp_bnn.compile(loss='MeanAbsoluteError',
                   optimizer=optimizers.Adam(lr=0.007),
                   metrics=['MeanAbsolutePercentageError',
                            'MeanAbsoluteError',
                            MAE_SBP, MAE_DBP, MAE_MBP])
elif loss_name == 'MAPE':
    bp_bnn.compile(loss='MeanAbsolutePercentageError',
                   optimizer=optimizers.Adam(lr=0.007),
                   metrics=['MeanAbsoluteError',
                            'MeanAbsolutePercentageError',
                            MAE_SBP, MAE_DBP, MAE_MBP])
else:
    bp_bnn.compile(loss=negative_loglikelihood,
                   ooptimizer=optimizers.RMSprop(learning_rate=0.007),
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
                             epochs=250,
                             batch_size=32,
                             verbose=2)
else:
    print('Training using CPU')
    history = bp_bnn.fit(X_train,
                         y_train,
                         epochs=250,
                         batch_size=32,
                         verbose=2)

print("Training done!")
prediction_distribution = bp_bnn(X_test)
print(prediction_distribution)

bp_bnn.save('./workspace/bp_estimation_BNN/model/bp_bnn_model_' + loss_name)

print("Evaluate on validation data")
results = bp_bnn.evaluate(X_val, y_val, batch_size=32)
print(results)
