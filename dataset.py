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

# Labels distribution
# [2.46622e+05, 1.33650e+04, 2.77949e+05, 4.99050e+04, 2.32000e+02, 4.39649e+05, 1.06033e+05, 1.44579e+05]
# [1.92924541e-01, 1.04550141e-02, 2.17430656e-01, 3.90390931e-02, 1.81486216e-04, 3.43923419e-01, 8.29462410e-02, 1.13099550e-01]

import numpy as np
from sklearn.model_selection import train_test_split


dataset_path = "dataset/cullpdb+profile_6133.npy"

sequence_len = 700
total_features = 57
amino_acid_residues = 21
num_classes = 8


cnn_width = 15


def get_dataset():
    ds = np.load(dataset_path)
    ds = np.reshape(ds, (ds.shape[0], sequence_len, total_features))
    ds = ds[:, :, 0:amino_acid_residues+ 1 + num_classes]  # remove unnecessary features
    return ds


def get_data_labels(D):
    X = D[:, :, 0:amino_acid_residues]
    Y = D[:, :, amino_acid_residues + 1:amino_acid_residues+ 1 + num_classes]
    return X, Y


def resphape_labels(labels):
    Y = np.reshape(labels, (labels.shape[0]*labels.shape[1], labels.shape[2]))
    Y = Y[~np.all(Y == 0, axis=1)]
    return Y


def reshape_data(X):
    # final shape should be (1278334, cnn_width, amino_acid_residues)
    padding = np.zeros((X.shape[0], X.shape[2], int(cnn_width/2)))
    X = np.dstack((padding, np.swapaxes(X, 1, 2), padding))
    X = np.swapaxes(X, 1, 2)
    # print("XShape after padding")
    # print(X.shape)
    res = np.zeros((X.shape[0], X.shape[1] - cnn_width + 1, cnn_width, amino_acid_residues))
    # print("res creation")
    # print(res.shape)
    for i in range(X.shape[1] - cnn_width + 1):
        res[:, i, :, :] = X[:, i:i+cnn_width, :]
    res = np.reshape(res, (X.shape[0]*(X.shape[1] - cnn_width + 1), cnn_width, amino_acid_residues))
    # print("res after for")
    # print(res.shape)
    res = res[res.sum(axis=2).sum(axis=1)>int(cnn_width/2), :, :]
    return res


def get_dataset_reshaped():
    D = get_dataset()
    X, Y = get_data_labels(D)
    X_reshaped = reshape_data(X)
    Y_reshaped = resphape_labels(Y)

    return X_reshaped, Y_reshaped


def split_dataset(X, Y, seed=None):
    X_tv, X_test, Y_tv, Y_test = train_test_split(X, Y, test_size=0.2, random_state=seed)
    X_train, X_validation, Y_train, Y_validation = train_test_split(X_tv, Y_tv, test_size=0.1, random_state=seed)
    return X_train, X_validation, X_test, Y_train, Y_validation, Y_test


if __name__ == '__main__':
    X, Y = get_dataset_reshaped()
    Y_dist = np.sum(Y, axis=0)
    print("Labels distribution")
    print(Y_dist)
    print(Y_dist / Y.shape[0])
    print("X shape")
    print(X.shape)
    print("Y shape")
    print(Y.shape)
    X_train, X_validation, X_test, Y_train, Y_validation, Y_test = split_dataset(X, Y)
    print("X_train shape")
    print(X_train.shape)
    print("X_validation shape")
    print(X_validation.shape)
    print("X_test shape")
    print(X_test.shape)
    print("Y_train shape")
    print(Y_train.shape)
    print("Y_validation shape")
    print(Y_validation.shape)
    print("Y_test shape")
    print(Y_test.shape)