import numpy as np

from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv1D, AveragePooling1D, MaxPooling1D, TimeDistributed, LeakyReLU, BatchNormalization
from keras import optimizers, callbacks
from keras.regularizers import l2
import keras.backend as K

import dataset

do_summary = True

LR = 0.002
drop_out = 0.4
batch_dim = 32
nn_epochs = 40

early_stop = callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=1, verbose=0, mode='min')

#filepath="NewModel-{epoch:02d}-{val_acc:.2f}.hdf5"
filepath="NewModel-best.hdf5"
checkpoint = callbacks.ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')

def CNN_model():
    # We fix the window size to 11 because the average length of an alpha helix is around eleven residues
    # and that of a beta strand is around six.
    # ref: https://www.researchgate.net/publication/285648102_Protein_Secondary_Structure_Prediction_Using_Deep_Convolutional_Neural_Fields
    m = Sequential()
    m.add(Conv1D(128, 11, padding='same', activation='relu', input_shape=(dataset.sequence_len, dataset.amino_acid_residues)))
    # m.add(BatchNormalization())
    # m.add(MaxPooling1D(pool_size=2, strides=1, padding='same'))
    m.add(Dropout(drop_out))
    m.add(Conv1D(64, 11, padding='same', activation='relu'))
    # m.add(BatchNormalization())
    # m.add(MaxPooling1D(pool_size=2, strides=1, padding='same'))
    m.add(Dropout(drop_out))
    # m.add(Conv1D(22, 11, padding='same', activation='relu'))
    # m.add(MaxPooling1D(pool_size=2, strides=1, padding='same'))
    # m.add(Dropout(drop_out))
    m.add(Conv1D(dataset.num_classes, 11, padding='same', activation='softmax'))

    opt = optimizers.Adam(lr=LR)
    m.compile(optimizer=opt,
              loss='categorical_crossentropy',
              metrics=['accuracy', 'mae'])
    if do_summary:
        print("\nHyper Parameters\n")
        print("Learning Rate: " + str(LR))
        print("Drop out: " + str(drop_out))
        print("Batch dim: " + str(batch_dim))
        print("Number of epochs: " + str(nn_epochs))

        m.summary()

    return m


def CNN_model_old():
    # We fix the window size to 11 because the average length of an alpha helix is around eleven residues
    # and that of a beta strand is around six.
    # ref: https://www.researchgate.net/publication/285648102_Protein_Secondary_Structure_Prediction_Using_Deep_Convolutional_Neural_Fields
    m = Sequential()
    m.add(Conv1D(64, 11, padding='same', activation='relu', input_shape=(dataset.sequence_len, dataset.amino_acid_residues)))
    m.add(MaxPooling1D(pool_size=5, strides=1, padding='same'))
    m.add(Dropout(drop_out))
    m.add(Conv1D(32, 11, padding='same', activation='relu'))
    m.add(MaxPooling1D(pool_size=5, strides=1, padding='same'))
    m.add(Dropout(drop_out))
    m.add(TimeDistributed(Dense(32, activation='relu')))
    m.add(MaxPooling1D(pool_size=2, strides=1, padding='same'))
    m.add(Dropout(drop_out))
    m.add(TimeDistributed(Dense(dataset.num_classes, activation='softmax', name='output')))
    opt = optimizers.Adam(lr=LR)
    m.compile(optimizer=opt,
              loss='categorical_crossentropy',
              metrics=['accuracy', 'mae'])
    
    if do_summary:
        print("\nHyper Parameters\n")
        print("Learning Rate: " + str(LR))
        print("Drop out: " + str(drop_out))
        print("Batch dim: " + str(batch_dim))
        print("Number of epochs: " + str(nn_epochs))

        m.summary()

    return m


if __name__ == '__main__':
    print("This script contains the model")
