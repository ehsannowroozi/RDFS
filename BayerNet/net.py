
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from keras.initializers import TruncatedNormal, Constant
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, BatchNormalization
import numpy as np
from lrn_layer import LRN2D
from keras import regularizers


def stamm_net(in_shape=(64, 64, 1), num_classes=2):

    model = Sequential()

    model.add(Conv2D(12, kernel_size=(5, 5), strides=(1, 1), input_shape=in_shape, name='convRes',
                     kernel_initializer=TruncatedNormal(0.0, 0.01), bias_initializer=Constant(0.1),
                     kernel_regularizer=regularizers.l2(0.001)))

    model.add(Conv2D(64, kernel_size=(7, 7), strides=(2, 2), name='conv1', activation='relu', padding='same',
                     kernel_initializer=TruncatedNormal(0.0, 0.01), bias_initializer=Constant(0.1),
                     kernel_regularizer=regularizers.l2(0.001)))

    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='pool1'))

    # model.add(LRN2D(alpha=0.0001, beta=0.75, n=5))
    model.add(BatchNormalization())

    model.add(Conv2D(48, kernel_size=(5, 5), strides=(1, 1), name='conv2', activation='relu', padding='same',
              kernel_initializer=TruncatedNormal(0.0, 0.01), bias_initializer=Constant(0.1),
              kernel_regularizer=regularizers.l2(0.001)))

    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='pool2'))
    # model.add(LRN2D(alpha=0.0001, beta=0.75, n=5))
    model.add(BatchNormalization())

    model.add(Flatten())
    model.add(Dense(4096, activation='relu',
                    kernel_initializer=TruncatedNormal(0.0, 0.005), bias_initializer=Constant(1),
                    kernel_regularizer=regularizers.l2(0.001)))
    model.add(Dropout(0.5))
    model.add(Dense(4096, activation='relu',
                    kernel_initializer=TruncatedNormal(0.0, 0.005), bias_initializer=Constant(1),
                    kernel_regularizer=regularizers.l2(0.001)))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax', name='predictions',
                    kernel_regularizer=regularizers.l2(0.001)))

    return model


def constrain_net_weights(model, p=1):

    # convRes layer has index 0
    weights = model.layers[0].get_weights()

    # make a copy of weights
    new_weights = weights

    # Kernel center
    c = int(np.floor(int(model.layers[0].kernel.shape[0])/2))

    # For each filter
    for k in range(weights[0].shape[-1]):

        wk = p * weights[0][:, :, 0, k]

        wk[c, c] = 0
        sum_wk = np.array(wk).sum()

        wk[:, :] /= sum_wk
        wk[c, c] = -1

        # Simple check: make sure weights sum up to 0
        # print(np.array(wk).sum())

        # Update filter weights and force biases to 0
        new_weights[0][:, :, 0, k] = wk
        new_weights[1][k] = 0

    # Put new weights back on place
    model.layers[0].set_weights(new_weights)

    return model
