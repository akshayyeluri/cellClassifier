# Keras, tf, and sklearn
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D, AvgPool2D
from keras.regularizers import l2


def model_1d(n_feat, conv_channels, dense_neurons, n_class = 10, act='relu', \
                             pool_type='avg', conv_ks = None, \
                             pool_ks = None, pads = None, \
                             opt='adam', metrics=['accuracy'], \
                             regs=0.01,
                             dropout_strength=0.5): 

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
    