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
from time import time
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv1D, AveragePooling1D, MaxPooling1D, TimeDistributed, LeakyReLU
from keras import optimizers, callbacks
from keras.regularizers import l2
import keras.backend as K

from timeit import default_timer as timer

do_log = False

LR = 0.001
drop_out = 0.4
batch_dim = 64
nn_epochs = 40

dataset_path = "dataset/cullpdb+profile_6133.npy"

sequence_len = 700
total_features = 57
amino_acid_residues = 22
num_classes = 9


def get_dataset():
    ds = np.load(dataset_path)
    ds = np.reshape(ds, (ds.shape[0], sequence_len, total_features))
    ds = ds[:, :, 0:amino_acid_residues + num_classes]  # remove unnecessary features
    return ds


def get_data_labels(D):
    X = D[:, :, 0:amino_acid_residues]
    Y = D[:, :, amino_acid_residues:amino_acid_residues + num_classes]
    return X, Y


def split_like_paper(Dataset):
    # Dataset subdivision following dataset readme and paper
    Train = Dataset[0:5600, :, :]
    Test = Dataset[5600:5877, :, :]
    Validation = Dataset[5877:, :, :]
    return Train, Test, Validation


def split_with_shuffle(Dataset, seed=None):
    np.random.seed(seed)
    np.random.shuffle(Dataset)
    return split_like_paper(Dataset)


def Q8_score(real, pred):
    total = real.shape[0] * real.shape[1]
    correct = 0
    for i in range(real.shape[0]):
        for j in range(real.shape[1]):
            if real[i,j,num_classes - 1] > 0:  # np.sum(real[i, j, :]) == 0
                total = total - 1
            else:
                if real[i, j, np.argmax(pred[i, j, :])] > 0:
                    correct = correct + 1

    return correct / total


def Q8_score_Tensor(y_true, y_pred):
    # test
    pass


def CNN_model():
    # We fix the window size to 11 because the average length of an alpha helix is around eleven residues
    # and that of a beta strand is around six.
    # ref: https://www.researchgate.net/publication/285648102_Protein_Secondary_Structure_Prediction_Using_Deep_Convolutional_Neural_Fields
    m = Sequential()
    m.add(Conv1D(64, 11, padding='same', activation='relu', input_shape=(sequence_len, amino_acid_residues)))
    m.add(MaxPooling1D(pool_size=5, strides=1, padding='same'))
    m.add(Dropout(drop_out))
    m.add(Conv1D(32, 11, padding='same', activation='relu'))
    m.add(MaxPooling1D(pool_size=5, strides=1, padding='same'))
    m.add(Dropout(drop_out))
    m.add(TimeDistributed(Dense(32, activation='relu')))
    m.add(MaxPooling1D(pool_size=2, strides=1, padding='same'))
    m.add(Dropout(drop_out))
    m.add(TimeDistributed(Dense(num_classes, activation='softmax', name='output')))
    opt = optimizers.Adam(lr=LR)
    m.compile(optimizer=opt,
              loss='categorical_crossentropy',
              metrics=['accuracy', 'mae'])
    m.summary()

    return m


print("Hyper Parameters\n")
print("Learning Rate: " + str(LR))
print("Drop out: " + str(drop_out))
print("Batch dim: " + str(batch_dim))
print("Number of epochs: " + str(nn_epochs))

start_time = timer()

dataset = get_dataset()

D_train, D_test, D_val = split_with_shuffle(dataset, 100)

X_train, Y_train = get_data_labels(D_train)
X_test, Y_test = get_data_labels(D_test)
X_val, Y_val = get_data_labels(D_val)

model = CNN_model()

early_stop = callbacks.EarlyStopping(monitor='val_loss', min_delta=0.005, patience=0, verbose=0, mode='min')

filepath="Model-{epoch:02d}-{val_acc:.2f}.hdf5"
checkpoint = callbacks.ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')

history = None

if do_log:
    history = model.fit(X_train, Y_train, epochs=nn_epochs, batch_size=batch_dim, shuffle=True,
                        validation_data=(X_val, Y_val), callbacks=[checkpoint,
            callbacks.TensorBoard(log_dir="logs/test/{}".format(time()), histogram_freq=1, write_graph=True),
            early_stop])
else:
    history = model.fit(X_train, Y_train, epochs=nn_epochs, batch_size=batch_dim, shuffle=True,
                        validation_data=(X_val, Y_val), callbacks=[checkpoint, early_stop])

predictions = model.predict(X_test)

print("\n\nQ8 accuracy:")
print(Q8_score(Y_test, predictions))

end_time = timer()

print("Time elapsed: " + str(end_time - start_time))

import matplotlib.pyplot as plt

plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

