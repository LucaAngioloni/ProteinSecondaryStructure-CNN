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
from keras import optimizers, callbacks
from timeit import default_timer as timer
from dataset import get_dataset, split_with_shuffle, get_data_labels, split_like_paper
import model
from time import time

do_log = True
stop_early = False
show_plots = True

dataset = get_dataset()

D_train, D_test, D_val = split_with_shuffle(dataset, 100)

X_train, Y_train = get_data_labels(D_train)
X_test, Y_test = get_data_labels(D_test)
X_val, Y_val = get_data_labels(D_val)

net = model.CNN_model()

start_time = timer()

history = None

call_b = [model.checkpoint]

if do_log:
    call_b.append(callbacks.TensorBoard(log_dir="../logs/Whole_CullPDB/{}".format(time()), histogram_freq=0, write_graph=True))

if stop_early:
    call_b.append(model.early_stop)

history = net.fit(X_train, Y_train, epochs=model.nn_epochs, batch_size=model.batch_dim, shuffle=True,
                        validation_data=(X_val, Y_val), callbacks=call_b)

end_time = timer()
print("\n\nTime elapsed: " + "{0:.2f}".format((end_time - start_time)) + " s")

predictions = net.predict(X_test)

print("\n\nQ8 accuracy: " + str(model.Q8_accuracy(Y_test, predictions)) + "\n\n")

if show_plots:
    from plot_history import plot_history
    plot_history(history)

