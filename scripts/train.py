#!/usr/bin/env python
# -*- coding: utf-8 -*-

#
# Authors: Dmitriy Serdyuk

from __future__ import print_function
import numpy
import argparse
import mimir

from contextlib import closing

from musicnet.callbacks import (
    SaveLastModel, Performance, Validation, LearningRateScheduler)
from musicnet.dataset import MusicNet
from musicnet.models.torch_models import DeepConvnet, train_model
from musicnet.visdom_handler import VisdomHandler


# input dimensions
d = 16384 / 4
window_size = d
# number of notes
m = 84
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


def get_model(model, feature_dim):
    if model.startswith('complex'):
        complex_ = True
        model = model.split('_')[1]
    else:
        complex_ = False
    if complex_:
        #model_module = models.complex
        print('.. complex network')
    else:
        #model_module = models
        pass
    if model == 'mlp':
        print('.. using MLP')
        return model_module.get_mlp(window_size=numpy.prod(feature_dim))
    elif model == 'shallow_convnet':
        print('.. using shallow convnet')
        return model_module.get_shallow_convnet(window_size=feature_dim[0],
                                                channels=feature_dim[1])
    elif model == 'deep_convnet':
        print('.. using deep convnet')
        return DeepConvnet(window_size=feature_dim[0],
                           channels=feature_dim[1])
    else:
        raise ValueError


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
    model = get_model(model, dataset.feature_dim)

    model.summary()
    #print(".. parameters: {:03.2f}M".format(model.count_params() / 1000000.))

    if in_memory:
        pass
        # do nothing
    else:
        raise ValueError

    it = dataset.train_iterator()

    print('.. start training')
    model.cuda()

    logger = mimir.Logger(
            filename='models/log_{}.jsonl.gz'.format(model_name))
    loss_handler = VisdomHandler(
        ['train', 'valid'], 'loss',
        dict(title='Train/valid cross-entropy',
             xlabel='iteration',
             ylabel='cross-entropy'), port=5004)
    ap_handler = VisdomHandler(
        ['valid'], 'ap',
        dict(title='Train/valid average precision',
             xlabel='iteration',
             ylabel='average precision'), port=5004)
    logger.handlers.extend([loss_handler, ap_handler])
    with closing(logger):
        train_model(dataset, model, steps_per_epoch=1000, epochs=epochs, 
                    cuda=True, logger=logger, lr_schedule=schedule)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('model_name')
    parser.add_argument('--in-memory', action='store_true', default=False)
    parser.add_argument('--complex', dest='complex_', action='store_true',
                        default=False)
    parser.add_argument('--model', default='shallow_convnet')
    parser.add_argument('--epochs', default=200, type=int)
    parser.add_argument('--fourier', action='store_true', default=False)
    parser.add_argument('--stft', action='store_true', default=False)
    parser.add_argument('--fast-load', action='store_true', default=False)
    parser.add_argument(
        '--local-data', 
        default="/Tmp/serdyuk/data/musicnet_11khz.npz")
    main(**parser.parse_args().__dict__)
