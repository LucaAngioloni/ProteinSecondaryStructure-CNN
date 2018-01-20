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
from keras.layers import Dense, Activation, Dropout, Conv1D, AveragePooling1D, MaxPooling1D, TimeDistributed, LeakyReLU, BatchNormalization
from keras import optimizers, callbacks
from keras.regularizers import l2
# import keras.backend as K
import tensorflow as tf

import dataset

do_summary = True

LR = 0.002
drop_out = 0.4
batch_dim = 32
nn_epochs = 40

#loss = 'categorical_hinge' # ok
loss = 'categorical_crossentropy' # best standart
#loss = 'mean_absolute_error' # bad
#loss = 'mean_squared_logarithmic_error' # new best (a little better)


early_stop = callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=1, verbose=0, mode='min')

#filepath="NewModel-{epoch:02d}-{val_acc:.2f}.hdf5"
filepath="NewModelConv-best.hdf5"
checkpoint = callbacks.ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')


def Q8_accuracy(real, pred):
    total = real.shape[0] * real.shape[1]
    correct = 0
    for i in range(real.shape[0]):  # per element in the batch
        for j in range(real.shape[1]): # per aminoacid residue
            if real[i, j, dataset.num_classes - 1] > 0:  # np.sum(real[i, j, :]) == 0  # if it is padding
                total = total - 1
            else:
                if real[i, j, np.argmax(pred[i, j, :])] > 0:
                    correct = correct + 1

    return correct / total


def CNN_model():
    m = Sequential()
    m.add(Conv1D(128, 5, padding='same', activation='relu', input_shape=(dataset.cnn_width, dataset.amino_acid_residues)))  # <----
    # m.add(BatchNormalization())
    # m.add(MaxPooling1D(pool_size=2, strides=1, padding='same'))
    m.add(Dropout(drop_out))  # <----
    m.add(Conv1D(64, 3, padding='same', activation='relu'))  # <----
    # m.add(BatchNormalization())
    m.add(MaxPooling1D(pool_size=2))
    m.add(Dropout(drop_out))  # <----
    m.add(Conv1D(32, 3, padding='same', activation='relu'))  # <----
    m.add(Flatten())
    m.add(Dense())
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

def CNN_model_old():
    # We fix the window size to 11 because the average length of an alpha helix is around eleven residues
    # and that of a beta strand is around six.
    # ref: https://www.researchgate.net/publication/285648102_Protein_Secondary_Structure_Prediction_Using_Deep_Convolutional_Neural_Fields
    m = Sequential()
    m.add(Conv1D(128, 11, padding='same', activation='relu', input_shape=(dataset.sequence_len, dataset.amino_acid_residues)))  # <----
    # m.add(BatchNormalization())
    # m.add(MaxPooling1D(pool_size=2, strides=1, padding='same'))
    m.add(Dropout(drop_out))  # <----
    m.add(Conv1D(64, 11, padding='same', activation='relu'))  # <----
    # m.add(BatchNormalization())
    # m.add(MaxPooling1D(pool_size=2, strides=1, padding='same'))
    m.add(Dropout(drop_out))  # <----
    # m.add(Conv1D(22, 11, padding='same', activation='relu'))
    # m.add(MaxPooling1D(pool_size=2, strides=1, padding='same'))
    # m.add(Dropout(drop_out))
    m.add(Conv1D(dataset.num_classes, 11, padding='same', activation='softmax'))  # <----
    # m.add(Conv1D(dataset.num_classes, 11, padding='same'))
    # m.add(TimeDistributed(Activation('softmax')))
    # m.add(Conv1D(dataset.num_classes, 11, padding='same', activation='softmax', input_shape=(dataset.sequence_len, dataset.amino_acid_residues)))
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
