import keras
from keras.models import Model
from keras.layers import (
    Input, BatchNormalization, Activation, Dense, Flatten)
from keras.layers.convolutional import Convolution1D, AveragePooling1D
from keras.regularizers import l2

from standard_resnet.vision_resnet import residual_block


def get_mlp(window_size=4096, output_size=84):
    model = keras.models.Sequential()
    model.add(keras.layers.Flatten(input_shape=(window_size, 1)))
    model.add(keras.layers.Dense(2048, activation='relu',
                                 kernel_initializer='glorot_normal'))
    model.add(keras.layers.Dense(output_size, activation='sigmoid',
                                 kernel_initializer='glorot_normal',
                                 bias_initializer=keras.initializers.Constant(value=0)))
    model.compile(optimizer=keras.optimizers.Adam(lr=1e-3),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model


def get_convnet(window_size=4096, output_size=84):
    model = keras.models.Sequential()
    model.add(keras.layers.Conv1D(
        64, 512, strides=4, input_shape=(window_size, 1),
        activation='linear',
        kernel_initializer='glorot_normal'))
    model.add(keras.layers.normalization.BatchNormalization(axis=-1))
    model.add(keras.layers.Activation('relu'))
    model.add(keras.layers.MaxPooling1D(pool_size=4, strides=2))

    model.add(keras.layers.Conv1D(
        32, 3, activation='relu',
        kernel_initializer='glorot_normal'))
    model.add(keras.layers.Conv1D(
        32, 3,
        activation='linear',
        kernel_initializer='glorot_normal'))
    model.add(keras.layers.normalization.BatchNormalization(axis=-1))
    model.add(keras.layers.Activation('relu'))

    #model.add(keras.layers.MaxPooling1D(pool_size=2))
    #model.add(keras.layers.Conv1D(64, 3,
    #                              kernel_initializer='glorot_normal'))
    #model.add(keras.layers.normalization.BatchNormalization(axis=-1))
    #model.add(keras.layers.Activation('relu'))

    model.add(keras.layers.MaxPooling1D(pool_size=8))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(2048, activation='relu',
                                 kernel_initializer='glorot_normal'))
    model.add(keras.layers.Dense(output_size, activation='sigmoid',
                                 bias_initializer=keras.initializers.Constant(value=0)))
    model.compile(optimizer=keras.optimizers.Adam(lr=1e-3),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model


def get_music_resnet(inp_shape=(4096, 1), spatial_drop_prob=0, output_size=128):
    '''This function returns the 50-layer residual network model
    you should load pretrained weights if you want to use it directly.
    Note that since the pretrained weights is converted from caffemodel
    the order of channels for input image should be 'BGR' (the channel order of caffe)
    '''

    inp = Input(shape=inp_shape)
    bn_axis = -1
    filsize = 3 

    out = Convolution1D(
        filters=8, kernel_size=512, strides=4, activation='linear', padding='same', 
        kernel_initializer='he_normal', use_bias=False, 
        kernel_regularizer=l2(0.0001), name='conv1')(inp)
    out = BatchNormalization(epsilon=1e-04, momentum=0.9, axis=bn_axis, name='bn_conv1')(out)
    out = Activation('relu')(out)

    num_blocks = 3
    for i in range(num_blocks):
        out = residual_block(
            input_tensor=out, filter_size=filsize, featmaps=[8, 8], stage=2, 
            strides=(4,),
            block='{}'.format(i), shortcut='regular', 
            spatial_drop_prob=spatial_drop_prob, dimensions=1, bn_axis=-1)

    for i in range(num_blocks):
        shortcut = 'projection' if i == 0 else 'regular'
        out = residual_block(
            input_tensor=out, filter_size=filsize, featmaps=[16, 16], stage=3,
            strides=(4,),
            block='{}'.format(i), shortcut=shortcut, spatial_drop_prob=spatial_drop_prob,
            dimensions=1, bn_axis=-1)

    for i in range(num_blocks):
        shortcut = 'projection' if i == 0 else 'regular'
        out = residual_block(
            input_tensor=out, filter_size=filsize, featmaps=[32, 32], stage=4,
            strides=(4,),
            block='{}'.format(i), shortcut=shortcut, spatial_drop_prob=spatial_drop_prob,
            dimensions=1, bn_axis=-1)

    out = AveragePooling1D(pool_size=64)(out)
    out = Flatten()(out)

    out = Dense(output_size, activation='sigmoid', kernel_regularizer=l2(0.0001),
                bias_initializer=keras.initializers.Constant(value=-5))(out)

    model = Model(inp, out)

    return model
