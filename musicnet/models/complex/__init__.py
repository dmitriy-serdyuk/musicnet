#!/usr/bin/env python
# -*- coding: utf-8 -*-

#
# Authors: Dmitriy Serdyuk, Olexa Bilaniuk, Chiheb Trabelsi, Sandeep Subramanian

import keras.backend as K
import keras
from keras.layers import Lambda, add, concatenate, Reshape, Concatenate
from keras.layers.convolutional import (
    Convolution2D, Convolution1D, MaxPooling1D, AveragePooling1D)
from keras.layers.core import Dense, Activation, Flatten
from keras.layers.normalization import BatchNormalization
from keras.initializers import Constant
from keras.optimizers import Adam
from keras.regularizers import l2
from keras.models import Model, Input
from keras.layers.core import Permute

from complexnn import (
    ComplexConv1D, ComplexBN, ComplexDense, GetReal, GetImag, GetAbs)


def get_shallow_convnet(window_size=4096, channels=2, output_size=84):
    inputs = Input(shape=(window_size, channels))

    conv = ComplexConv1D(
        32, 512, strides=16,
        activation='relu')(inputs)
    pool = AveragePooling1D(pool_size=4, strides=2)(conv)

    pool = Permute([2, 1])(pool)
    flattened = Flatten()(pool)

    dense = ComplexDense(2048, activation='relu')(flattened)
    predictions = ComplexDense(
        output_size, 
        activation='sigmoid',
        bias_initializer=Constant(value=-5))(dense)
    predictions = GetReal(predictions)
    model = Model(inputs=inputs, outputs=predictions)

    model.compile(optimizer=Adam(lr=1e-4),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model


def get_deep_convnet(window_size=4096, channels=2, output_size=84):
    inputs = Input(shape=(window_size, channels))
    outs = inputs

    outs = (ComplexConv1D(
        16, 6, strides=2, padding='same',
        activation='linear',
        kernel_initializer='complex_independent'))(outs)
    outs = (ComplexBN(axis=-1))(outs)
    outs = (keras.layers.Activation('relu'))(outs)
    outs = (keras.layers.AveragePooling1D(pool_size=2, strides=2))(outs)

    outs = (ComplexConv1D(
        32, 3, strides=2, padding='same',
        activation='linear',
        kernel_initializer='complex_independent'))(outs)
    outs = (ComplexBN(axis=-1))(outs)
    outs = (keras.layers.Activation('relu'))(outs)
    outs = (keras.layers.AveragePooling1D(pool_size=2, strides=2))(outs)
    
    outs = (ComplexConv1D(
        64, 3, strides=1, padding='same',
        activation='linear',
        kernel_initializer='complex_independent'))(outs)
    outs = (ComplexBN(axis=-1))(outs)
    outs = (keras.layers.Activation('relu'))(outs)
    outs = (keras.layers.AveragePooling1D(pool_size=2, strides=2))(outs)

    outs = (ComplexConv1D(
        64, 3, strides=1, padding='same',
        activation='linear',
        kernel_initializer='complex_independent'))(outs)
    outs = (ComplexBN(axis=-1))(outs)
    outs = (keras.layers.Activation('relu'))(outs)
    outs = (keras.layers.AveragePooling1D(pool_size=2, strides=2))(outs)

    outs = (ComplexConv1D(
        128, 3, strides=1, padding='same',
        activation='relu',
        kernel_initializer='complex_independent'))(outs)
    outs = (ComplexConv1D(
        128, 3, strides=1, padding='same',
        activation='linear',
        kernel_initializer='complex_independent'))(outs)
    outs = (ComplexBN(axis=-1))(outs)
    outs = (keras.layers.Activation('relu'))(outs)
    outs = (keras.layers.AveragePooling1D(pool_size=2, strides=2))(outs)

    #outs = (keras.layers.MaxPooling1D(pool_size=2))
    #outs = (Permute([2, 1]))
    outs = (keras.layers.Flatten())(outs)
    outs = (keras.layers.Dense(2048, activation='relu',
                           kernel_initializer='glorot_normal'))(outs)
    predictions = (keras.layers.Dense(output_size, activation='sigmoid',
                                 bias_initializer=keras.initializers.Constant(value=-5)))(outs)

    model = Model(inputs=inputs, outputs=predictions)
    model.compile(optimizer=keras.optimizers.Adam(lr=1e-4),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model



def learn_concat_real_imag_block(
        input_tensor, filter_size, featmaps,
        stage, block, activation='relu', bn_axis=None, dimensions=2):
    if dimensions == 2:
        convolution = Convolution2D
    else:
        convolution = Convolution1D

    nb_fmaps1, nb_fmaps2 = featmaps
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    channels_ordering = K.image_data_format()
    if bn_axis is None:
        if channels_ordering == 'channels_first':
            channel_axis = 1
        elif channels_ordering == 'channels_last':
            channel_axis = -1
        else:
            raise ValueError('Invalid image_data_format:', channels_ordering)
    else:
        channel_axis = bn_axis
    
    out = BatchNormalization(
        epsilon=1e-04,
        momentum=0.9,
        axis=channel_axis,
        name=bn_name_base + '2a'
    )(input_tensor)

    out = Activation(activation)(out)

    out = convolution(
        nb_fmaps1,
        filter_size,
        padding='same',
        kernel_initializer='he_normal',
        use_bias=False,
        kernel_regularizer=l2(0.0001),
        name=conv_name_base + '2a'
    )(out)
    
    out = BatchNormalization(
        epsilon=1e-04,
        momentum=0.9,
        axis=channel_axis,
        name=bn_name_base + '2b'
    )(out)

    out = Activation(activation)(out)

    out = convolution(
        nb_fmaps2,
        filter_size,
        padding='same',
        kernel_initializer='he_normal',
        use_bias=False,
        kernel_regularizer=l2(0.0001),
        name=conv_name_base + '2b'
    )(out)

    return out


def get_residual_block(I, filter_size, featmaps, stage, block, shortcut, 
                       convArgs, bnArgs, d):
    """Get residual block."""
    
    activation           = d.act
    drop_prob            = d.dropout
    nb_fmaps1, nb_fmaps2 = featmaps
    conv_name_base       = 'res'+str(stage)+block+'_branch'
    bn_name_base         = 'bn' +str(stage)+block+'_branch'
    if K.image_data_format() == 'channels_first' and K.ndim(I) != 3:
        channel_axis = 1
    else:
        channel_axis = -1
    
    if   d.model == "real":
        O = BatchNormalization(name=bn_name_base+'_2a', **bnArgs)(I)
    elif d.model == "complex":
        O = ComplexBN(name=bn_name_base+'_2a', **bnArgs)(I)
    O = Activation(activation)(O)
    
    if shortcut == 'regular' or d.spectral_pool_scheme == "nodownsample":
        if   d.model == "real":
            O = Conv2D(nb_fmaps1, filter_size, name=conv_name_base+'2a', **convArgs)(O)
        elif d.model == "complex":
            O = ComplexConv2D(nb_fmaps1, filter_size, name=conv_name_base+'2a', **convArgs)(O)
    elif shortcut == 'projection':
        if d.spectral_pool_scheme == "proj":
            O = applySpectralPooling(O, d)
        if   d.model == "real":
            O = Conv2D(nb_fmaps1, filter_size, name=conv_name_base+'2a', strides=(2, 2), **convArgs)(O)
        elif d.model == "complex":
            O = ComplexConv2D(nb_fmaps1, filter_size, name=conv_name_base+'2a', strides=(2, 2), **convArgs)(O)
    
    if   d.model == "real":
        O = BatchNormalization(name=bn_name_base+'_2b', **bnArgs)(O)
        O = Activation(activation)(O)
        O = Conv2D(nb_fmaps2, filter_size, name=conv_name_base+'2b', **convArgs)(O)
    elif d.model == "complex":
        O = ComplexBN(name=bn_name_base+'_2b', **bnArgs)(O)
        O = Activation(activation)(O)
        O = ComplexConv2D(nb_fmaps2, filter_size, name=conv_name_base+'2b', **convArgs)(O)
    
    if   shortcut == 'regular':
        O = Add()([O, I])
    elif shortcut == 'projection':
        if d.spectral_pool_scheme == "proj":
            I = applySpectralPooling(I, d)
        if   d.model == "real":
            X = Conv2D(nb_fmaps2, (1, 1),
                       name    = conv_name_base+'1',
                       strides = (2, 2) if d.spectral_pool_scheme != "nodownsample" else
                                 (1, 1),
                       **convArgs)(I)
            O = Concatenate(channel_axis)([X, O])
        elif d.model == "complex":
            X = ComplexConv2D(nb_fmaps2, (1, 1),
                              name    = conv_name_base+'1',
                              strides = (2, 2) if d.spectral_pool_scheme != "nodownsample" else
                                        (1, 1),
                              **convArgs)(I)
            
            O_real = Concatenate(channel_axis)([GetReal(X), GetReal(O)])
            O_imag = Concatenate(channel_axis)([GetImag(X), GetImag(O)])
            O = Concatenate(      1     )([O_real,     O_imag])
    
    return O


def get_music_resnet(inp_shape=(4096, 1), spatial_drop_prob=0, output_size=128):
    '''This function returns the 50-layer residual network model
    you should load pretrained weights if you want to use it directly.
    Note that since the pretrained weights is converted from caffemodel
    the order of channels for input image should be 'BGR' (the channel order of caffe)
    '''

    inp = Input(shape=inp_shape)
    bn_axis = -1
    filsize = 3 

    out = ComplexConvolution1D(
        filters=2, kernel_size=5, strides=2, activation='linear', padding='same', 
        kernel_initializer='he_normal', use_bias=False, 
        kernel_regularizer=l2(0.0001), name='conv1')(inp)
    out = ComplexBN(epsilon=1e-04, momentum=0.9, axis=bn_axis, name='bn_conv1')(out)
    out = Activation('relu')(out)

    num_blocks = 3
    for i in range(num_blocks):
        out = get_residual_block(
            input_tensor=out, filter_size=filsize, featmaps=[2, 2], stage=2, 
            strides=(1,), 
            block='{}'.format(i), shortcut='regular', 
            spatial_drop_prob=spatial_drop_prob, dimensions=1, bn_axis=-1)

    for i in range(num_blocks):
        shortcut = 'projection' if i == 0 else 'regular'
        stride = 2 if i == 0 else 1
        out = get_residual_block(
            input_tensor=out, filter_size=filsize, featmaps=[4, 4], stage=3,
            strides=(stride,),
            block='{}'.format(i), shortcut=shortcut, spatial_drop_prob=spatial_drop_prob,
            dimensions=1, bn_axis=-1)

    for i in range(num_blocks):
        shortcut = 'projection' if i == 0 else 'regular'
        stride = 2 if i == 0 else 1
        out = get_residual_block(
            input_tensor=out, filter_size=filsize, featmaps=[8, 8], stage=4,
            strides=(stride,),
            block='{}'.format(i), shortcut=shortcut, spatial_drop_prob=spatial_drop_prob,
            dimensions=1, bn_axis=-1)

    for i in range(num_blocks):
        shortcut = 'projection' if i == 0 else 'regular'
        stride = 2 if i == 0 else 1
        out = get_residual_block(
            input_tensor=out, filter_size=filsize, featmaps=[16, 16], stage=5,
            strides=(stride,),
            block='{}'.format(i), shortcut=shortcut, spatial_drop_prob=spatial_drop_prob,
            dimensions=1, bn_axis=-1)

    for i in range(num_blocks):
        shortcut = 'projection' if i == 0 else 'regular'
        stride = 2 if i == 0 else 1
        out = get_residual_block(
            input_tensor=out, filter_size=filsize, featmaps=[32, 32], stage=6,
            strides=(stride,),
            block='{}'.format(i), shortcut=shortcut, spatial_drop_prob=spatial_drop_prob,
            dimensions=1, bn_axis=-1)

    for i in range(num_blocks):
        shortcut = 'projection' if i == 0 else 'regular'
        stride = 2 if i == 0 else 1
        out = get_residual_block(
            input_tensor=out, filter_size=filsize, featmaps=[64, 64], stage=7,
            strides=(stride,),
            block='{}'.format(i), shortcut=shortcut, spatial_drop_prob=spatial_drop_prob,
            dimensions=1, bn_axis=-1)

    for i in range(num_blocks):
        shortcut = 'projection' if i == 0 else 'regular'
        stride = 2 if i == 0 else 1
        out = get_residual_block(
            input_tensor=out, filter_size=filsize, featmaps=[128, 128], stage=8,
            strides=(stride,),
            block='{}'.format(i), shortcut=shortcut, spatial_drop_prob=spatial_drop_prob,
            dimensions=1, bn_axis=-1)

    out = AveragePooling1D(pool_size=32)(out)
    out = Flatten()(out)

    out = Dense(output_size, activation='sigmoid', kernel_regularizer=l2(0.0001),
                bias_initializer=keras.initializers.Constant(value=-5))(out)

    model = Model(inp, out)
    sgd = keras.optimizers.SGD(lr=0.1, momentum=0.9, decay=0, nesterov=True, clipnorm=1)

    model.compile(optimizer=sgd, loss='binary_crossentropy', 
                  metrics=['accuracy'])

    return model
