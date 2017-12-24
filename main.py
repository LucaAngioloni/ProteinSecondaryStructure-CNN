import numpy as np
from time import time
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv1D, AveragePooling1D, MaxPooling1D, TimeDistributed
from keras import optimizers, callbacks

do_log = False

LR = 1e-3
drop_out = 0.4
batch_dim = 64
nn_epochs = 10

dataset_path = "dataset/cullpdb+profile_6133.npy"

sequence_len = 700
total_features = 57
amino_acid_residues = 22
num_classes = 9


def Q8_score(real, pred):
    total = real.shape[0]*real.shape[1]
    correct = 0
    for i in range(real.shape[0]):
        for j in range(real.shape[1]):
            if real[i,j,num_classes-1] > 0:
                total = total - 1
            else:
                if real[i,j,np.argmax(pred[i,j,:])] > 0:
                    correct = correct + 1

    return correct/total

def CNN_model():
    m = Sequential()
    m.add(Conv1D(128, 5, padding='same', activation='relu', input_shape=(sequence_len, amino_acid_residues)))
    m.add(AveragePooling1D(pool_size=2, strides=1, padding='same'))
    m.add(Dropout(drop_out))
    m.add(Conv1D(128, 5, padding='same', activation='relu'))
    m.add(AveragePooling1D(pool_size=2, strides=1, padding='same'))
    m.add(Dropout(drop_out))
    m.add(Conv1D(64, 3, padding='same', activation='relu'))
    m.add(AveragePooling1D(pool_size=4, strides=1, padding='same'))
    m.add(Dropout(drop_out))
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

model = CNN_model()

if do_log:
    model.fit(X_train, Y_train, epochs=nn_epochs, batch_size=batch_dim, shuffle=True, validation_data=(X_val, Y_val),
          callbacks=[
              callbacks.TensorBoard(log_dir="logs/test/{}".format(time()), histogram_freq=1, write_graph=True)])
else:
    model.fit(X_train, Y_train, epochs=nn_epochs, batch_size=batch_dim, shuffle=True, validation_data=(X_val, Y_val))

predictions = model.predict(X_test)

print("\n\nQ8 accuracy:")
print(Q8_score(Y_test, predictions))
