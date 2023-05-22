from keras.layers import Dense, BatchNormalization
from keras.layers import Activation
from keras.layers import Dropout
from keras.models import Sequential


def create_bp_ann(input_dim, activation, num_class):
    model = Sequential()

    model.add(Dense(35, input_dim=input_dim))
    model.add(Activation(activation))

    model.add(Dense(20))
    model.add(Activation(activation))

    model.add(Dense(num_class))

    return model
