from __future__ import print_function
import sys
import os
import time

import keras
from keras import backend as K
from keras.layers import Dense, Flatten, Activation, Conv2D, MaxPooling2D, Dropout
from keras.losses import binary_crossentropy, mean_squared_error
from keras.models import Sequential, Model

from sklearn.model_selection import train_test_split

import numpy
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

from itertools import cycle

from datetime import datetime

from batchDataset import Dataset
from deepexplain.tensorflow import DeepExplain

total_start_time = datetime.now()

batch_size = 100
num_classes = 2
num_features = 21000
epochs = 1

dataFile = "/home/myc/projectpy/DeepFam/data/COG-500-1074/90percent/data388to1_withanother2.txt"

CHARSET = {'A': 0, 'C': 1, 'D': 2, 'E': 3, 'F': 4, 'G': 5, 'H': 6, \
           'I': 7, 'K': 8, 'L': 9, 'M': 10, 'N': 11, 'P': 12, 'Q': 13, \
           'R': 14, 'S': 15, 'T': 16, 'V': 17, 'W': 18, 'Y': 19, 'X': 20, \
           'O': 20, 'U': 20,
           'B': 20,
           'Z': 20,
           'J': 20}
CHARLEN = 21


def numerate_seq(seq_raw):
    arr = numpy.zeros((1, 1000), dtype=numpy.int16)

    for j, c in enumerate(seq_raw):
        if j == 1000:
            continue
        if c == '\n':
            continue
        if c == "_":
            # let them zero
            continue
        elif isinstance(CHARSET[c], int):
            idx = CHARSET[c]
            arr[0][j] = idx
        else:
            raise Exception("not reach here")
    return arr


def encoding_seq_np(seq_raw):
    arr = numpy.zeros((len(seq_raw), CHARLEN * 1000), dtype=numpy.float32)
    for i, seq in enumerate(seq_raw):
        for j, c in enumerate(seq):
            if j == 1000:
                continue
            else:
                idx = CHARLEN * j + c
                arr[i][idx] = 1
    return arr


def get_Data():
    X = []
    Y = []

    print("data File:", dataFile)

    for index, line in enumerate(open(dataFile, 'r').readlines()):
        w = line.split('\t')
        label = w[0]
        features = w[1]
        # if index ==99999:
        #     print("haha")
        # print("data line:", index)

        try:
            label = [int(x) for x in label]
            features = numerate_seq(features)
        except ValueError:
            print('Line %s is corrupt!' % index)
            break

        X.append(features)
        Y.extend(label)

    X = numpy.asarray(X)
    Y = numpy.asarray(Y)

    # print("feature shape:", X.shape)
    print("label shape:", Y.shape)

    return X, Y


X, Y = get_Data()

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, random_state=42)

x_train = X_train
# x_train = numpy.reshape(x_train, (len(x_train), 1, 21000, 1))
ky_train = y_train

x_test = X_test
# x_test = numpy.reshape(x_test, (len(x_test), 1, 21000, 1))

print("training data count:", len(x_train))

y_train = keras.utils.to_categorical(ky_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

img_x, img_y = 1, 21000

input_shape = (img_x, img_y, 1)

model = Sequential()
model.add(Conv2D(48, kernel_size=3, strides=(1, 1), padding='same',
                 activation='relu',
                 input_shape=input_shape))

# model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
# model.add(MaxPooling2D(pool_size=(1, 1)))

model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes))
model.add(Activation('softmax'))
model.summary()

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])


class AccuracyHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}): self.acc = []

    def on_epoch_end(self, batch, logs={}): self.acc.append(logs.get('acc'))


history = AccuracyHistory()

train_data = Dataset(x_train, y_train)
total_batch = int(train_data._num_examples / batch_size)

x_test = numpy.reshape(x_test, (len(x_test), 1000))
x_test = encoding_seq_np(x_test)
x_test = numpy.reshape(x_test, (len(x_test), 1, 21000, 1))

avg_cost = 0
avg_acc = 0

for i in range(total_batch):
    batch_x_raw, batch_y = train_data.next_batch(batch_size)
    batch_x = numpy.reshape(batch_x_raw, (len(batch_x_raw), 1000))
    batch_x = encoding_seq_np(batch_x)
    batch_x = numpy.reshape(batch_x, (len(batch_x), 1, 21000, 1))

    model.fit(batch_x, batch_y,
              batch_size=batch_size,
              epochs=epochs,
              verbose=1,
              validation_data=(x_test, y_test))

score = model.evaluate(x_test, y_test, verbose=0, batch_size=batch_size)

print('Test loss:', score[0])
print('Test accuracy:', score[1])

with DeepExplain(session=K.get_session()) as de:  # <-- init DeepExplain context
    # Need to reconstruct the graph in DeepExplain context, using the same weights.
    # With Keras this is very easy:
    # 1. Get the input tensor to the original model
    input_tensor = model.layers[0].input

    # 2. We now target the output of the last dense layer (pre-softmax)
    # To do so, create a new model sharing the same layers untill the last dense (index -2)
    fModel = Model(inputs=input_tensor, outputs=model.layers[-2].output)
    fModel.summary()
    target_tensor = fModel(input_tensor)

    xs = x_test[0:10]
    ys = y_test[0:10]

    attributions = de.explain('grad*input', target_tensor * ys, input_tensor, xs)
    # attributions = de.explain('saliency', target_tensor * ys, input_tensor, xs)
    # attributions = de.explain('intgrad', target_tensor * ys, input_tensor, xs)
    # attributions = de.explain('deeplift', target_tensor * ys, input_tensor, xs)
    # attributions = de.explain('elrp', target_tensor * ys, input_tensor, xs)
    # attributions = de.explain('occlusion', target_tensor * ys, input_tensor, xs)

    relvance = numpy.reshape(attributions, (len(attributions), 1000, 21))
    relvance = relvance.mean(axis=2)

    # Plot attributions
    from utils import plot, plt

    n_cols = 4
    n_rows = int(len(attributions) / 2)
    fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(3 * n_cols, 3 * n_rows))

    for i, a in enumerate(attributions):
        row, col = divmod(i, 2)
        plot(xs[i].reshape(1000, 1), cmap='Greys', axis=axes[row, col * 2]).set_title('Original')
        #plot(a.reshape(1000, 1), xi=relvance[i], axis=axes[row, col * 2 + 1]).set_title('Attributions')
    plt.show()
