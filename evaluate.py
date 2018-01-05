import numpy as np
from time import time
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv1D, AveragePooling1D, MaxPooling1D, TimeDistributed, LeakyReLU
from keras import optimizers, callbacks
from keras.regularizers import l2
import keras.backend as K

from timeit import default_timer as timer

do_log = False

LR = 0.002
drop_out = 0.4
batch_dim = 32
nn_epochs = 50

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
              metrics=['accuracy'])
    m.summary()

    return m


def CNN_model_new():
    # We fix the window size to 11 because the average length of an alpha helix is around eleven residues
    # and that of a beta strand is around six.
    # ref: https://www.researchgate.net/publication/285648102_Protein_Secondary_Structure_Prediction_Using_Deep_Convolutional_Neural_Fields
    m = Sequential()
    m.add(Conv1D(88, 11, padding='same', activation='relu', input_shape=(sequence_len, amino_acid_residues)))
    m.add(Dropout(drop_out))
    m.add(Conv1D(44, 11, padding='same', activation='relu'))
    m.add(Dropout(drop_out))
    m.add(Conv1D(num_classes, 11, padding='same', activation='softmax'))

    opt = optimizers.Adam(lr=LR)
    m.compile(optimizer=opt,
              loss='categorical_crossentropy',
              metrics=['accuracy'])
    m.summary()

    return m

dataset = get_dataset()

D_train, D_test, D_val = split_with_shuffle(dataset, 100)

X_train, Y_train = get_data_labels(D_train)
X_test, Y_test = get_data_labels(D_test)
X_val, Y_val = get_data_labels(D_val)

model = CNN_model_new()

#load Weights
model.load_weights("NewModel-best.hdf5")

predictions = model.predict(X_test)

print("\n\nQ8 accuracy:")
print(Q8_score(Y_test, predictions))

end_time = timer()

print("Time elapsed: " + str(end_time - start_time))
