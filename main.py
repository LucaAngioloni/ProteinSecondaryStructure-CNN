import numpy as np
from time import time
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv1D, AveragePooling1D, MaxPooling1D, TimeDistributed
from keras import optimizers, callbacks
from keras.regularizers import l2

from timeit import default_timer as timer

do_log = False

LR = 0.001
drop_out = 0.5
batch_dim = 128
nn_epochs = 4

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

def split_with_shuffle(Dataset, seed = None):
    np.random.seed(seed)
    np.random.shuffle(Dataset)
    return split_like_paper(Dataset)

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

def Q8_score_Tensor(real, pred):
    pass

def CNN_model():
    m = Sequential()
    m.add(Conv1D(128, 5, padding='same', activation='relu', input_shape=(sequence_len, amino_acid_residues)))
    m.add(MaxPooling1D(pool_size=5, strides=1, padding='same'))
    m.add(Dropout(drop_out))
    m.add(Conv1D(64, 5, padding='same', activation='relu'))
    m.add(MaxPooling1D(pool_size=5, strides=1, padding='same'))
    m.add(Dropout(drop_out))
    m.add(Conv1D(64, 3, padding='same', activation='relu'))
    m.add(MaxPooling1D(pool_size=3, strides=1, padding='same'))
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

start_time = timer()

dataset = get_dataset()

D_train, D_test, D_val = split_with_shuffle(dataset, 10)

X_train, Y_train = get_data_labels(D_train)
X_test, Y_test = get_data_labels(D_test)
X_val, Y_val = get_data_labels(D_val)

model = CNN_model()

early_stop = callbacks.EarlyStopping(monitor='val_loss', min_delta=0.05, patience=0, verbose=0, mode='min')

if do_log:
    model.fit(X_train, Y_train, epochs=nn_epochs, batch_size=batch_dim, shuffle=True, validation_data=(X_val, Y_val),
          callbacks=[
              callbacks.TensorBoard(log_dir="logs/test/{}".format(time()), histogram_freq=1, write_graph=True), early_stop])
else:
    model.fit(X_train, Y_train, epochs=nn_epochs, batch_size=batch_dim, shuffle=True, validation_data=(X_val, Y_val), callbacks=[early_stop])

predictions = model.predict(X_test)

print("\n\nQ8 accuracy:")
print(Q8_score(Y_test, predictions))

end_time = timer()

print("Time elapsed: " + str(end_time - start_time))

