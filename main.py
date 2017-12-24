import numpy as np
from time import time
from keras.models import Sequential
from keras.models import Model
from keras.layers import Dense, Activation, Dropout, Conv1D, Convolution1D, Input, AveragePooling1D, MaxPooling1D, TimeDistributed
from keras import optimizers, callbacks

LR = 1e-3
drop_out = 0.3
batch_dim = 20
nn_epochs = 10

dataset_path = "dataset/cullpdb+profile_6133.npy"

sequence_len = 700
total_features = 57
amino_acid_residues = 22
num_classes = 9

def Q8_score(pred, real):
    np.mean(np.mean(np.abs(pred-real)))

def CNN_model():
    m = Sequential()
    m.add(Conv1D(20, 5, padding='same', activation='relu', input_shape=(sequence_len, amino_acid_residues)))
    m.add(AveragePooling1D(pool_size=2, strides=1, padding='same'))
    m.add(Conv1D(20, 3, padding='same', activation='relu'))
    m.add(AveragePooling1D(pool_size=2, strides=1, padding='same'))
    m.add(TimeDistributed(Dense(num_classes, activation='softmax', name='output')))
    opt = optimizers.Adam(lr=LR)
    m.compile(optimizer=opt,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    m.summary()

    return m


dataset = np.load(dataset_path)
dataset = np.reshape(dataset, (dataset.shape[0], sequence_len, total_features))
dataset = dataset[:, :, 0:amino_acid_residues + num_classes]  # remove unnecessary features

# shuffle maybe?

X = dataset[:, :, 0:amino_acid_residues]
Y = dataset[:, :, amino_acid_residues:amino_acid_residues + num_classes]

# Dataset subdivision following dataset readme and paper
X_train = X[0:5600, :, :]
Y_train = Y[0:5600, :, :]

X_test = X[5600:5877, :, :]
Y_test = Y[5600:5877, :, :]

X_val = X[5877:, :, :]
Y_val = Y[5877:, :, :]


print(X_train.shape)
print(Y_train.shape)

model = CNN_model()

model.fit(X_train, Y_train, epochs=nn_epochs, batch_size=batch_dim, shuffle=True, validation_data=(X_val, Y_val),
          callbacks=[
              callbacks.TensorBoard(log_dir="logs/test/{}".format(time()), histogram_freq=1, write_graph=True)])

pred = model.predict(X_test)


print(Q8_score(pred, Y_test))