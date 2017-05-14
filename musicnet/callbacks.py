import os
import numpy as np
from os import path
from sklearn.metrics import precision_recall_curve, average_precision_score

from keras.callbacks import Callback, ModelCheckpoint, LearningRateScheduler


class SaveLastModel(Callback):
    def __init__(self, workdir, period=10, name=''):
        self.name = name
        self.workdir = workdir
        self.chkptsdir = path.join(self.workdir, "chkpts")

        if not path.isdir(self.chkptsdir):
            os.mkdir(self.chkptsdir)

        self.period_of_epochs = period
        self.link_filename = path.join(self.chkptsdir, 
                                          "model_checkpoint.hdf5")
    
    def on_epoch_end(self, epoch, logs={}):
        if (epoch + 1) % self.period_of_epochs == 0:
            # Filenames
            base_hdf5_filename = "model{}_checkpoint{:06d}.hdf5".format(
                self.name, epoch + 1)
            base_yaml_filename = "model{}_checkpoint{:06d}.yaml".format(
                self.name, epoch + 1)
            hdf5_filename = path.join(self.chkptsdir, base_hdf5_filename)
            yaml_filename = path.join(self.chkptsdir, base_yaml_filename)
            
            # YAML
            yaml_model = self.model.to_yaml()
            with open(yaml_filename, "w") as yaml_file:
                yaml_file.write(yaml_model)
            
            # HDF5
            keras.models.save_model(self.model, hdf5_filename)
            with h5py.File(hdf5_filename, "r+") as f:
                f.require_dataset("initialEpoch", (), "uint64", True)[...] = int(epoch+1)
                f.flush()
            
            # Symlink to new HDF5 file, then atomically rename and replace.
            os.symlink(base_hdf5_filename, self.link_filename + ".rename")
            os.rename (self.link_filename + ".rename",
                       self.link_filename)


class Performance(Callback):
    def __init__(self, logger, **kwargs):
        self.logger = logger
        super(self, Performance).__init__(**kwargs)

    def on_epoch_begin(self, epoch, logs={}):
        self.timestamps = []

    def on_epoch_end(self, epoch, logs={}):
        t = np.asarray(self.timestamps, dtype=np.float64)
        train_function_time = float(np.mean(t[ :,1] - t[:  ,0]))
        load_data_time = float(np.mean(t[1:,0] - t[:-1,1]))
        self.logger.log(
            {'epoch': epoch, 'train_function_time': train_function_time})
        self.logger.log({'epoch': epoch, 'load_data_time': load_data_time})

    def on_batch_begin(self, epoch, logs={}):
        self.timestamps += [[time(), time()]]

    def on_batch_end(self, epoch, logs={}):
        self.timestamps[-1][-1] = time()


class Validation(Callback):
    def __init__(self, x, y, name, logger):
        self.x = x
        self.y = y
        self.name = name
        self.logger = logger

    def on_train_begin(epochs, logs):
        pr = self.model.predict(self.x[:, :, None])
        average_precision  = average_precision_score(
            self.y.flatten(), pr.flatten())
        self.logger.log(
            {'epoch': epoch, self.name + "_avg_precision": average_precision})

    def on_epoch_end(epoch, logs):
        pr = self.model.predict(self.x[:, :, None])
        average_precision = average_precision_score(
            self.x.flatten(), pr.flatten())
        self.logger.log(
            {'epoch': epoch, self.name + "_avg_precision": average_precision})

