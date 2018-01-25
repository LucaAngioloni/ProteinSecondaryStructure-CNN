# MIT License
#
# Copyright (c) 2017 Luca Angioloni
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import numpy as np

from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Conv1D, AveragePooling1D, MaxPooling1D, TimeDistributed, LeakyReLU, BatchNormalization, Flatten
from keras import optimizers, callbacks
from keras.regularizers import l2
# import keras.backend as K
import tensorflow as tf

import dataset

do_summary = True

LR = 0.0009 # maybe after some (10-15) epochs reduce it to 0.0008-0.0007
drop_out = 0.38
batch_dim = 64
nn_epochs = 35

#loss = 'categorical_hinge' # ok
loss = 'categorical_crossentropy' # best standart
#loss = 'mean_absolute_error' # bad
#loss = 'mean_squared_logarithmic_error' # new best (a little better)


early_stop = callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=1, verbose=0, mode='min')

#filepath="NewModel-{epoch:02d}-{val_acc:.2f}.hdf5"
if dataset.filtered:
    filepath="CullPDB_Filtered-best.hdf5"
else:
    filepath="CullPDB6133-best.hdf5"

checkpoint = callbacks.ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')


def CNN_model():
    m = Sequential()
    m.add(Conv1D(128, 5, padding='same', activation='relu', input_shape=(dataset.cnn_width, dataset.amino_acid_residues)))
    m.add(BatchNormalization())
    # m.add(MaxPooling1D(pool_size=2))
    m.add(Dropout(drop_out))
    m.add(Conv1D(128, 3, padding='same', activation='relu'))
    m.add(BatchNormalization())
    # m.add(MaxPooling1D(pool_size=2))
    m.add(Dropout(drop_out))
    m.add(Conv1D(64, 3, padding='same', activation='relu'))
    m.add(BatchNormalization())
    # m.add(MaxPooling1D(pool_size=2))
    m.add(Dropout(drop_out))
    # m.add(Conv1D(32, 3, padding='same', activation='relu'))
    # m.add(BatchNormalization())
    # m.add(MaxPooling1D(pool_size=2))
    # m.add(Dropout(drop_out))
    m.add(Flatten())
    m.add(Dense(128, activation='relu'))
    m.add(Dense(32, activation='relu'))
    m.add(Dense(dataset.num_classes, activation = 'softmax'))
    opt = optimizers.Adam(lr=LR)
    m.compile(optimizer=opt,
              loss=loss,
              metrics=['accuracy', 'mae'])
    if do_summary:
        print("\nHyper Parameters\n")
        print("Learning Rate: " + str(LR))
        print("Drop out: " + str(drop_out))
        print("Batch dim: " + str(batch_dim))
        print("Number of epochs: " + str(nn_epochs))
        print("\nLoss: " + loss + "\n")

        m.summary()

    return m


if __name__ == '__main__':
    print("This script contains the model")
