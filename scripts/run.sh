#!/usr/bin/env bash

. $HOME/.bashrc
pushd ..
. ./env.sh
popd

train.py <model_name> --fourier --in-memory --model=deep_convnet --complex