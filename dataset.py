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
import os

filtered = True

if filtered:
    dataset_path = "dataset/cullpdb+profile_6133_filtered.npy"
else:
    dataset_path = "dataset/cullpdb+profile_6133.npy"

cb513_path = "dataset/cb513+profile_split1.npy"

sequence_len = 700
total_features = 57
amino_acid_residues = 21
num_classes = 8

cnn_width = 17

##
## @brief      Determines if filtered dataset is used.
##
## @return     True if filtered, False otherwise.
##
def is_filtered():
    return filtered

##
## @brief      Gets the dataset in the original form from path.
##
## @param      path  The path
##
## @return     The dataset, numpy array.
##
def get_dataset(path="dataset/cullpdb+profile_6133.npy"):
    ds = np.load(path)
    ds = np.reshape(ds, (ds.shape[0], sequence_len, total_features))
    ret = np.zeros((ds.shape[0], ds.shape[1], amino_acid_residues + num_classes))
    ret[:, :, 0:amino_acid_residues] = ds[:, :, 35:56]
    ret[:, :, amino_acid_residues:] = ds[:, :, amino_acid_residues + 1:amino_acid_residues+ 1 + num_classes]
    return ret

##
## @brief      Gets the labels from dataset split.
##
## @param      D     Dataset split
##
## @return     The labels.
##
def get_data_labels(D):
    X = D[:, :, 0:amino_acid_residues]
    Y = D[:, :, amino_acid_residues:amino_acid_residues + num_classes]
    return X, Y


##
## @brief      Reshapes the lables (700,8,Len) to (8,Len*700)
##
## @param      labels  The labels
##
## @return     The Labels reshaped
##
def resphape_labels(labels):
    Y = np.reshape(labels, (labels.shape[0]*labels.shape[1], labels.shape[2]))
    Y = Y[~np.all(Y == 0, axis=1)]
    return Y

##
## @brief      Creates new dataset from the original shifting a window of cnn_width len on the dataset sequences
##
## @param      X     The Dataset features, numpy array
##
## @return     The dataset reshaped
##
def reshape_data(X):
    padding = np.zeros((X.shape[0], X.shape[2], int(cnn_width/2)))
    X = np.dstack((padding, np.swapaxes(X, 1, 2), padding))
    X = np.swapaxes(X, 1, 2)
    res = np.zeros((X.shape[0], X.shape[1] - cnn_width + 1, cnn_width, amino_acid_residues))
    for i in range(X.shape[1] - cnn_width + 1):
        res[:, i, :, :] = X[:, i:i+cnn_width, :]
    res = np.reshape(res, (X.shape[0]*(X.shape[1] - cnn_width + 1), cnn_width, amino_acid_residues))
    res = res[np.count_nonzero(res, axis=(1,2))>(int(cnn_width/2)*amino_acid_residues), :, :]
    return res

##
## @brief      Gets the dataset in the reshaped form.
##
## @param      seed  Random seeed for the split
##
## @return     The dataset.
##
def get_dataset_reshaped(seed=None):
    D = get_dataset(dataset_path)
    Train, Test, Validation = split_dataset(D, seed)
    X_te, Y_te = get_data_labels(Test)
    X_tr, Y_tr = get_data_labels(Train)
    X_v, Y_v = get_data_labels(Validation)

    X_train = reshape_data(X_tr)
    X_test = reshape_data(X_te)
    X_validation = reshape_data(X_v)

    Y_train = resphape_labels(Y_tr)
    Y_test = resphape_labels(Y_te)
    Y_validation = resphape_labels(Y_v)

    return X_train, X_validation, X_test, Y_train, Y_validation, Y_test

##
## @brief      Splits the dataset.
##
## @param      Dataset  The dataset
## @param      seed     Random seeed for the split
##
## @return     Returns Train, Test, Validation tensors from the Dataset
##
def split_dataset(Dataset, seed=None):
    np.random.seed(seed)
    np.random.shuffle(Dataset)
    train_split = int(Dataset.shape[0]*0.8)
    test_val_split = int(Dataset.shape[0]*0.1)
    Train = Dataset[0:train_split, :, :]
    Test = Dataset[train_split:train_split+test_val_split, :, :]
    Validation = Dataset[train_split+test_val_split:, :, :]
    return Train, Test, Validation

##
## @brief      Splits the dataset with the same subdivision as the paper: Jian Zhou and Olga G. Troyanskaya (2014) - "Deep Supervised and Convolutional Generative Stochastic Network for Protein Secondary Structure Prediction"
##
## @param      Dataset  The dataset
## @param      seed     Random seeed for the split
##
## @return     Returns Train, Test, Validation tensors from the Dataset
##
def split_like_paper(Dataset, seed=None):
    # Dataset subdivision following dataset readme and paper
    if seed is not None:
        np.random.seed(seed)
        np.random.shuffle(Dataset)
    Train = Dataset[0:5600, :, :]
    Test = Dataset[5600:5877, :, :]
    Validation = Dataset[5877:, :, :]
    return Train, Test, Validation

##
## @brief      Gets the dataset in the reshaped form splitted like in the paper.
##
## @return     The resphaped dataset.
##
def get_resphaped_dataset_paper():
    D = get_dataset()
    Train, Test, Validation = split_like_paper(D)
    X_te, Y_te = get_data_labels(Test)
    X_tr, Y_tr = get_data_labels(Train)
    X_v, Y_v = get_data_labels(Validation)

    X_train = reshape_data(X_tr)
    X_test = reshape_data(X_te)
    X_validation = reshape_data(X_v)

    Y_train = resphape_labels(Y_tr)
    Y_test = resphape_labels(Y_te)
    Y_validation = resphape_labels(Y_v)

    return X_train, X_validation, X_test, Y_train, Y_validation, Y_test

##
## @brief      Gets the CB513 dataset.
##
## @return     The CB513 dataset.
##
def get_cb513():
    CB = get_dataset(cb513_path)
    X, Y = get_data_labels(CB)
    return reshape_data(X), resphape_labels(Y)


if __name__ == '__main__':
    print("Collectiong dataset...")
    D = get_dataset()
    X, Y = get_data_labels(D)
    Y_dist = np.sum(Y, axis=0)
    print("Labels distribution")
    print(Y_dist)
    print(Y_dist / Y.shape[0])
    print("X shape")
    print(X.shape)
    print("Y shape")
    print(Y.shape)