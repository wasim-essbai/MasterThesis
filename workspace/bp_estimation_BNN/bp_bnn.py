from keras.layers import Input
from keras.layers import Dense
from keras.layers import Dropout
from keras.models import Sequential
from keras import layers
import tensorflow_probability as tfp
import tensorflow as tf

FEATURE_NAMES = [
    "cp",
    "sut",
    "dt",
    "dt10",
    "st10",
    "st10_p_dt10",
    "st10_d_dt10",
    "dt25",
    "st25",
    "st25_p_dt25",
    "st25_d_dt25",
    "dt33",
    "st33",
    "st33_p_dt33",
    "st33_d_dt33",
    "dt50",
    "st50",
    "st50_p_dt50",
    "st50_d_dt50",
    "dt66",
    "st66",
    "st66_p_dt66",
    "st66_d_dt66",
    "dt75",
    "st75",
    "st75_p_dt75",
    "st75_d_dt75",
]


def create_model_inputs():
    inputs = {}
    for feature_name in FEATURE_NAMES:
        inputs[feature_name] = layers.Input(
            name=feature_name, shape=(1,), dtype=tf.float32
        )
    return inputs


def prior(kernel_size, bias_size, dtype=None):
    n = kernel_size + bias_size
    prior_model = Sequential(
        [
            tfp.layers.DistributionLambda(
                lambda t: tfp.distributions.MultivariateNormalDiag(
                    loc=tf.random.normal([n], 50, 80, tf.float32), 
                    scale_diag=tf.ones(n)*10
                )
            )
        ]
    )
    return prior_model


def posterior(kernel_size, bias_size, dtype=None):
    n = kernel_size + bias_size
    posterior_model = Sequential(
        [
            tfp.layers.VariableLayer(
                tfp.layers.MultivariateNormalTriL.params_size(n), dtype=dtype
            ),
            tfp.layers.MultivariateNormalTriL(n),
        ]
    )
    return posterior_model


def create_bp_bnn(input_dim, activation, train_size, num_class):
    model = Sequential()

    model.add(Input(input_dim))

    model.add(tfp.layers.DenseVariational(
        units=15,
        make_prior_fn=prior,
        make_posterior_fn=posterior,
        kl_weight=1 / train_size,
        activation=activation,
    ))
    # model.add(Dropout(0.5))

    model.add(tfp.layers.DenseVariational(
        units=6,
        make_prior_fn=prior,
        make_posterior_fn=posterior,
        kl_weight=1 / train_size,
        activation=activation,
    ))
    # model.add(Dropout(0.5))

    model.add(Dense(num_class*2))
    model.add(tfp.layers.IndependentNormal(num_class))

    return model
