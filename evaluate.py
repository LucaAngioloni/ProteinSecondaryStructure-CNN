import numpy as np
from keras import optimizers, callbacks
from timeit import default_timer as timer
from dataset import get_dataset, split_with_shuffle, get_data_labels, Q8_score, split_like_paper
import model

dataset = get_dataset()

D_train, D_test, D_val = split_with_shuffle(dataset, 100)

X_train, Y_train = get_data_labels(D_train)
X_test, Y_test = get_data_labels(D_test)
X_val, Y_val = get_data_labels(D_val)

model = model.CNN_model()

#load Weights
model.load_weights("Best Models/NewModel-best-05890-acc08791.hdf5")

predictions = model.predict(X_test)

print("\n\nQ8 accuracy:")
print(Q8_score(Y_test, predictions))

