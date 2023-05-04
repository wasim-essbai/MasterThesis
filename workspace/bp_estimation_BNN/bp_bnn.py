from keras.layers import Input
from keras.layers import Dense
from keras.layers import Dropout
from keras.models import Sequential
import tensorflow_probability as tfp
import tensorflow as tf


def prior(kernel_size, bias_size, dtype=None):
    n = kernel_size + bias_size
    prior_model = Sequential(
        [
            tfp.layers.DistributionLambda(
                lambda t: tfp.distributions.MultivariateNormalDiag(
                    loc=tf.zeros(n), scale_diag=tf.ones(n)
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

    model.add(Input(input_dim, name='FeatureInput'))

    model.add(tfp.layers.DenseVariational(
        units=35,
        make_prior_fn=prior,
        make_posterior_fn=posterior,
        kl_weight=1 / train_size,
        activation=activation,
    ))
    model.add(Dropout(0.5))

    model.add(tfp.layers.DenseVariational(
        units=20,
        make_prior_fn=prior,
        make_posterior_fn=posterior,
        kl_weight=1 / train_size,
        activation=activation,
    ))
    model.add(Dropout(0.5))

    model.add(Dense(2))
    model.add(tfp.layers.IndependentNormal(1))

    return model
