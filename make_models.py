'''
This module creates compiled keras models, given a bunch of input parameters;
It is useful for hyperparameter optimization using grid search.
'''

# Keras, tf, and sklearn
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D, AvgPool2D
from keras.regularizers import l2


def model_2_hidden_layer(n_feat, neurons_1, neurons_2, \
                         act_1='relu', act_2='relu', \
                         dropout_1=None, dropout_2=None, \
                         reg_1=0.01, reg_2=0.01, \
                         n_class=10, opt='adam', metrics=['accuracy']):
    """
    Makes a 2 hidden layer simple multilayer perceptron (all dense layers)
    model, with arguments for number of neurons in each hidden layer,
    as well as activation functions / l2 regularization strength / dropout
    strength for each layer
    """

    model = Sequential()
    # Hidden layer 1
    model.add(Dense(neurons_1, activation=act_1, \
                    input_shape=(n_feat,), \
                    kernel_regularizer=l2(reg_1), \
                    bias_regularizer=l2(reg_1)))
    if dropout_1 is not None:
        model.add(Dropout(dropout_1))

    # Hidden layer 2
    model.add(Dense(neurons_2, activation=act_2, \
                    kernel_regularizer=l2(reg_2), \
                    bias_regularizer=l2(reg_2)))
    if dropout_2 is not None:
        model.add(Dropout(dropout_2))

    # Final softmax layer
    model.add(Dense(n_class, activation='softmax'))

    model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=metrics)
    return model



def model_1d(n_feat, conv_channels, dense_neurons, n_class = 10, act='relu', \
                             pool_type='avg', conv_ks = None, \
                             pool_ks = None, pads = None, \
                             opt='adam', metrics=['accuracy'], \
                             regs=0.01,
                             dropout_strength=0.5): 
    
    """
    Makes a CNN model
    """

    n_conv, n_dense = len(conv_channels), len(dense_neurons)
    if conv_ks is None:
        conv_ks = [5] * n_conv
    if pool_ks is None:
        pool_ks = [2] * n_conv
    if pads is None:
        pads = ['same'] * n_conv
    if isinstance(act, str):
        act = [act] * (n_conv + n_dense)
    if isinstance(regs, float):
        regs = [regs] * (n_dense)
    Pool = MaxPooling2D if pool_type == 'max' else AvgPool2D


    model = Sequential()
    for i in range(n_conv):
        if i == 0:
            model.add(Convolution2D(conv_channels[i], (conv_ks[i], 1), activation=act[i], \
                      input_shape=(n_feat, 1, 1), padding = pads[i]))
        else:
            model.add(Convolution2D(conv_channels[i], (conv_ks[i], 1), \
                                    activation=act[i], padding=pads[i]))
        model.add(Pool(pool_size=(pool_ks[i], 1)))
       

    model.add(Flatten())
    for i in range(n_dense):
        model.add(Dense(dense_neurons[i], activation=act[i + n_conv], \
                        kernel_regularizer=l2(regs[i]), \
                        bias_regularizer=l2(regs[i])))
    model.add(Dropout(dropout_strength))
    model.add(Dense(n_class, activation='softmax'))

    model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=metrics)
    return model































    
                         
    
