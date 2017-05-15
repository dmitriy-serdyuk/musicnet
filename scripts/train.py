#!/usr/bin/env python
from __future__ import print_function
import numpy as np
import h5py
import os
from os import path
import sys
import argparse
import mimir

from time import time

import keras

from musicnet.callbacks import (
    SaveLastModel, Performance, Validation, LearningRateScheduler)
from musicnet.dataset import (
    create_test_in_memory, load_in_memory, music_net_iterator)
from musicnet import models


#d = 2048        # input dimensions
d = 16384        # input dimensions
d = d / 4
window_size = d
m = 84         # number of notes
fs = 44100      # samples/second
fs_resample = 11000
features = 0    # first element of (X,Y) data tuple
labels = 1      # second element of (X,Y) data tuple
step = 512
step = step / 4


def schedule(epoch):
    if epoch >= 0 and epoch < 10:
        lrate = 1e-4
        if epoch == 0:
            print('\ncurrent learning rate value is ' + str(lrate))
    elif epoch >= 10 and epoch < 100:
        lrate = 1e-4
        if epoch == 10:
            print('\ncurrent learning rate value is ' + str(lrate))
    elif epoch >= 100 and epoch < 120:
        lrate = 5e-5
        if epoch == 100:
            print('\ncurrent learning rate value is ' + str(lrate))
    elif epoch >= 120 and epoch < 150:
        lrate = 1e-5
        if epoch == 120:
            print('\ncurrent learning rate value is ' + str(lrate))
    elif epoch >= 150:
        lrate = 1e-6
        if epoch == 150:
            print('\ncurrent learning rate value is ' + str(lrate))
    return lrate


def get_model(model, complex_):
    if model == 'mlp':
        print('.. using MLP')
        return models.get_mlp()
    elif model == 'shallow_convnet':
        if complex_:
            return models.complex_models.get_shallow_convnet()
        print('.. using convnet')
        return models.get_shallow_convnet()
    elif model == 'deep_convnet':
        print('.. using convnet')
        return models.get_deep_convnet()
    else:
        raise ValueError
        if complex_:
            pass
        else:
            #model = get_music_resnet()
            
            # load YAML and create model
            with open('models/chkpts/model_checkpoint000010.yaml', 'r') as yaml_file:
                loaded_model_yaml = yaml_file.read()

            loaded_model = keras.models.model_from_yaml(loaded_model_yaml)
            # load weights into new model
            loaded_model.load_weights("models/chkpts/model_checkpoint000010.hdf5")
            print(".. loaded model from disk")
            model = loaded_model
        sgd = keras.optimizers.SGD(lr=0.1, momentum=0.9, decay=0, nesterov=True, clipnorm=1)

        model.compile(optimizer=sgd, loss='binary_crossentropy', 
                      metrics=['accuracy'])
        return model


def main(model_name, in_memory, complex_, model, local_data, epochs):
    print(".. building model")
    model = get_model(model, complex_)

    model.summary()
    print(".. parameters: {:03.2f}M".format(model.count_params() / 1000000.))

    # Warning: the full dataset is over 40GB. Make sure you have enough RAM!
    # This can take a few minutes to load
    if in_memory:
        print('.. loading train data')
        train_data, valid_data, test_data = load_in_memory(local_data)
        print('.. train data loaded')
        Xtest, Ytest = create_test_in_memory(test_data)
        Xvalid, Yvalid = create_test_in_memory(valid_data)
    else:
        raise ValueError

    if in_memory:
        pass
        # do nothing
    elif path.isfile('/Tmp/serdyuk/data/musicnet_test.npz'):
        print('!! reading cached test file')
        with open('/Tmp/serdyuk/data/musicnet_test.npz', 'rb') as f:
            contents = np.load(f)
            Xtest = contents['xtest']
            Ytest = contents['ytest']
            Xvalid = contents['xvalid']
            Yvalid = contents['yvalid']
    else:
        Xtest, Ytest = create_test(test_files, music_file, window_size=d, 
                                   resample=11000, step=step, 
                                   note_to_class=note_to_class,
                                   output_dim=m)
        Xvalid, Yvalid = create_test(valid_files, music_file, window_size=d,
                                     resample=11000, step=step, 
                                     note_to_class=note_to_class,
                                     output_dim=m)

        with open('/Tmp/serdyuk/data/musicnet_test.npz', 'wb') as f:
            np.savez(f, xtest=Xtest, ytest=Ytest, xvalid=Xvalid, yvalid=Yvalid)

    logger = mimir.Logger(
        filename='models/log_{}.jsonl.gz'.format(model_name))

    rng = np.random.RandomState(123)
    if in_memory: 
        it = music_net_iterator(train_data, rng)
    else:
        raise ValueError

    callbacks = [Validation(Xvalid, Yvalid, 'valid', logger), 
                 Validation(Xtest, Ytest, 'test', logger),
                 SaveLastModel("./models/", 1, name=model), 
                 Performance(logger),
                 LearningRateScheduler(schedule)
                 ]

    print('.. start training')
    model.fit_generator(
        it, steps_per_epoch=1000, epochs=epochs,
        callbacks=callbacks, workers=1
        #verbose=2
        #initial_epoch=10
       )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('model_name')
    parser.add_argument('--in-memory', action='store_true', default=False)
    parser.add_argument('--complex', dest='complex_', action='store_true', default=False)
    parser.add_argument('--model', default='resnet')
    parser.add_argument('--epochs', default=200, type=int)
    parser.add_argument(
        '--local-data', 
        default="/Tmp/serdyuk/data/musicnet_11khz.npz")
    main(**parser.parse_args().__dict__)