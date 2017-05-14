from .complex_convolution import Complex_Convolution1D as ComplexConvolution1D
from .complex_batch_normalization import (
    Complex_BatchNormalization as ComplexBatchNormalization)
from .resnet_models import get_imagpart, get_realpart, getpart_output_shape

import keras.backend as K
from keras.layers import Lambda, add, concatenate, Reshape
from keras.layers.convolutional import (
    AveragePooling2D, Convolution2D, AveragePooling3D, Convolution1D)
from keras.layers.core import Dense, Activation, Flatten
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2


def complex_residual_block(input_tensor, filter_size, featmaps, stage, block,
                           shortcut='regular', activation='relu',
                           drop_prob=0.2, dimensions=2, bn_axis=None):
    nb_fmaps1, nb_fmaps2 = featmaps
    if dimensions == 2:
        raise ValueError
    else:
        convolution = ComplexConvolution1D

    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    channels_ordering = K.image_data_format()
    if bn_axis is None:
        if channels_ordering == 'channels_first' and K.ndim(input_tensor) != 3:
            channel_axis = 1
        else:
            channel_axis = -1
    else:
        channel_axis = bn_axis

    out = ComplexBatchNormalization(
        epsilon=1e-04, momentum=0.9, axis=channel_axis,
        name=bn_name_base + '_2a'
    )(input_tensor)

    out = Activation(activation)(out)

    if shortcut == 'regular':
        out = convolution(
            nb_fmaps1,
            filter_size,
            padding='same',
            use_bias=False,
            kernel_regularizer=l2(0.0001),
            name=conv_name_base + '2a'
        )(out)
    elif shortcut == 'projection':
        out = convolution(
            nb_fmaps1,
            filter_size,
            padding='same',
            use_bias=False,
            strides=(2, 2),
            kernel_regularizer=l2(0.0001),
            name=conv_name_base + '2a'
        )(out)

    # out = Real_Imag_Dropout(drop_prob)(out)

    out = ComplexBatchNormalization(
        epsilon=1e-04, momentum=0.9, axis=channel_axis,
        name=bn_name_base + '_2b'
    )(out)

    out = Activation(activation)(out)

    out = convolution(
        nb_fmaps2,
        filter_size,
        padding='same',
        use_bias=False,
        kernel_regularizer=l2(0.0001),
        name=conv_name_base + '2b'
    )(out)

    # out = Real_Imag_Dropout(drop_prob)(out)

    if shortcut == 'regular':
        out = add([out, input_tensor])
    elif shortcut == 'projection':
        filter_size_projection = (1, 1) if dimensions == 2 else (1,)
        x = convolution(
            nb_fmaps2,
            filter_size_projection,
            use_bias=False,
            strides=(2, 2),
            kernel_regularizer=l2(0.0001),
            name=conv_name_base + '1'
        )(input_tensor)

        out_real = Lambda(get_realpart, output_shape=getpart_output_shape)(out)
        x_real = Lambda(get_realpart, output_shape=getpart_output_shape)(x)
        out_real = concatenate([x_real, out_real], axis=channel_axis)
        
        out_imag = Lambda(get_imagpart, output_shape=getpart_output_shape)(out)
        x_imag = Lambda(get_imagpart, output_shape=getpart_output_shape)(x)
        out_imag = concatenate([x_imag, out_imag], axis=channel_axis)

        out = concatenate([out_real, out_imag], axis=1)
    
    return out


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
