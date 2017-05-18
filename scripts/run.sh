#!/usr/bin/env bash

# Dataset: /data/lisatmp3/speech/musicnet/musicnet_11khz.npz
. $HOME/.bashrc
pushd ..
. ./env.sh
popd

train.py <model_name> --fourier --in-memory --model=deep_convnet --complex