#!/usr/bin/env python
from __future__ import print_function
import numpy
import numpy as np
from os import path
import argparse
import mimir

import keras

import musicnet.models.complex
from musicnet.callbacks import (
    SaveLastModel, Performance, Validation, LearningRateScheduler)
from musicnet.dataset import MusicNet
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
        lrate = 1e-3
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


def get_model(model, complex_, feature_dim):
    if complex_:
        model_module = models.complex
        print('.. complex network')
    else:
        model_module = models
    if model == 'mlp':
        print('.. using MLP')
        return model_module.get_mlp(window_size=numpy.prod(feature_dim))
    elif model == 'shallow_convnet':
        print('.. using shallow convnet')
        return model_module.get_shallow_convnet(window_size=feature_dim[0],
                                                channels=feature_dim[1])
    elif model == 'deep_convnet':
        print('.. using deep convnet')
        return model_module.get_deep_convnet(window_size=feature_dim[0],
                                             channels=feature_dim[1])
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


def main(model_name, in_memory, complex_, model, local_data, epochs, fourier,
         stft, fast_load):
    rng = numpy.random.RandomState(123)

    # Warning: the full dataset is over 40GB. Make sure you have enough RAM!
    # This can take a few minutes to load
    if in_memory:
        print('.. loading train data')
        dataset = MusicNet(local_data, complex_=complex_, fourier=fourier,
                           stft=stft, rng=rng, fast_load=fast_load)
        dataset.load()
        print('.. train data loaded')
        Xvalid, Yvalid = dataset.eval_set('valid')
        Xtest, Ytest = dataset.eval_set('test')
    else:
        raise ValueError

    print(".. building model")
    model = get_model(model, complex_, dataset.feature_dim)

    model.summary()
    print(".. parameters: {:03.2f}M".format(model.count_params() / 1000000.))

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

    it = dataset.train_iterator()

    callbacks = [Validation(Xvalid, Yvalid, 'valid', logger),
                 Validation(Xtest, Ytest, 'test', logger),
                 SaveLastModel("./models/", 1, name=model), 
                 Performance(logger),
                 LearningRateScheduler(schedule)]

    print('.. start training')
    model.fit_generator(
        it, steps_per_epoch=1000, epochs=epochs,
        callbacks=callbacks, workers=1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('model_name')
    parser.add_argument('--in-memory', action='store_true', default=False)
    parser.add_argument('--complex', dest='complex_', action='store_true',
                        default=False)
    parser.add_argument('--model', default='resnet')
    parser.add_argument('--epochs', default=200, type=int)
    parser.add_argument('--fourier', action='store_true', default=False)
    parser.add_argument('--stft', action='store_true', default=False)
    parser.add_argument('--fast-load', action='store_true', default=False)
    parser.add_argument(
        '--local-data', 
        default="/Tmp/serdyuk/data/musicnet_11khz.npz")
    main(**parser.parse_args().__dict__)
