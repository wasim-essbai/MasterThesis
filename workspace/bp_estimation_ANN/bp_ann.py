from keras.layers import Dense
from keras.layers import Activation
from keras.layers import Dropout
from keras.models import Sequential


def create_bp_ann(input_dim, activation, num_class):
    model = Sequential()

    model.add(Dense(1024, input_dim=input_dim))
    model.add(Activation(activation))
    model.add(Dropout(0.5))

    model.add(Dense(512))
    model.add(Activation(activation))
    model.add(Dropout(0.5))

    model.add(Dense(64))
    model.add(Activation(activation))
    model.add(Dropout(0.25))

    model.add(Dense(num_class))
    model.add(Activation('linear'))

    return model
