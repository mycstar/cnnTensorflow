from __future__ import print_function
import sys
import os
import time

import pandas
import numpy as np
from collections import Counter
import h5py
from keras.models import model_from_json

np.random.seed(1337)  # for reproducibility

import keras
from keras import backend as K
from keras import metrics
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM, GRU, SimpleRNN
from keras.layers.convolutional import Convolution1D, MaxPooling1D
from keras.datasets import imdb

from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit, GroupKFold
from datetime import datetime
from imblearn.over_sampling import RandomOverSampler
from imblearn.combine import SMOTEENN
from imblearn.over_sampling import SMOTE, ADASYN
from random import shuffle
from batchDataset import Dataset


# import cPickle


def encoding_seq_np(seq):
    arr = np.zeros((1, CHARLEN * 1000), dtype=np.float32)
    for i, c in enumerate(seq):
        if i == 1000:
            continue
        if c == '\n':
            continue
        if c == "_":
            # let them zero
            continue
        elif isinstance(CHARSET[c], int):
            idx = CHARLEN * i + CHARSET[c]
            arr[0][idx] = 1
        else:
            idx1 = CHARLEN * i + CHARSET[c][0]
            idx2 = CHARLEN * i + CHARSET[c][1]
            arr[0][idx1] = 0.5
            arr[0][idx2] = 0.5
            # raise Exception("notreachhere")
    return arr


def get_Data(dataFile):
    X = []
    Y = []
    # dataFile = "/home/myc/projectpy/cnnTensorflowNew/data/CaricaPapaya/caricapapaya_label_0_1000.txt"
    # dataFile = '/home/myc/projectpy/cnnTensorflowNew/data/CaricaPapaya/train_8000.txt'
    print("data File:", dataFile)

    for index, line in enumerate(open(dataFile, 'r').readlines()):
        w = line.split(',')
        label = w[0]
        features = w[1]
        # if index ==99999:
        #     print("haha")
        # print("data line:", index)

        try:
            label = [int(x) for x in label]
            features = encoding_seq_np(features)
        except ValueError:
            print('Line %s is corrupt!' % index)
            break

        X.append(features)
        Y.extend(label)

    X = np.asarray(X)
    X = np.reshape(X, (len(X), 21000))

    Y = np.asarray(Y)

    print("feature shape:", X.shape)
    print("label shape:", Y.shape)

    return X, Y


def get_Data_group1(dataFile):
    # dataFile = "/home/myc/projectpy/cnnTensorflowNew/data/CaricaPapaya/CA_group_features_CA_25800.fa.csv"
    print("data File:", dataFile)

    groupData = pandas.read_csv(dataFile, sep=",", header=0)
    groupData["sequence"] = groupData["sequence"].astype(str)
    print("total raw row :", len(groupData))
    groupData = groupData[(groupData['sequence'].str.len() > 50) & (groupData['sequence'].str.len() < 1000)]
    print("after remove invalid sequence <50 and > 1000, left row :", len(groupData))
    groupData['label'] = 1

    return groupData


def shuffle_list(*ls):
    l = list(zip(*ls))

    shuffle(l)
    return zip(*l)


def encoding_seq_np_list(seq, arr):
    for i, c in enumerate(seq):
        if i == 999:
            print("999")
        if c == "_":
            # let them zero
            continue
        elif isinstance(CHARSET[str.upper(c)], int):
            idx = CHARLEN * i + CHARSET[str.upper(c)]
            arr[idx] = 1
        else:
            idx1 = CHARLEN * i + CHARSET[str.upper(c)][0]
            idx2 = CHARLEN * i + CHARSET[str.upper(c)][1]
            arr[idx1] = 0.5
            arr[idx2] = 0.5
            # raise Exception("notreachhere")


def groupFilter(df, num):
    groups = pandas.Series(df["family"].tolist()).unique()
    print("total different group is:", len(groups))
    group_series = df.groupby(['family'])['label'].sum()
    ret_family_list = group_series[group_series >= num].keys().tolist()
    print("%s group memebers number are greater than %s" % (len(ret_family_list), num))
    print("group detail:", group_series[group_series >= num])

    ret_df = df.loc[df['family'].isin(ret_family_list)]
    ret_df = ret_df.groupby(['family']).head(num)
    print("select %s samples per group, total samples is:%s" % (num, len(ret_df)))

    return ret_df


def trans(str1):
    a = []
    dic = {'A': 1, 'B': 22, 'U': 23, 'J': 24, 'Z': 25, 'O': 26, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6, 'H': 7, 'I': 8,
           'K': 9, 'L': 10, 'M': 11, 'N': 12, 'P': 13, 'Q': 14, 'R': 15, 'S': 16, 'T': 17, 'V': 18, 'W': 19, 'Y': 20,
           'X': 21}
    for i in range(len(str1)):
        a.append(dic.get(str1[i]))
    return a


def createTrainData(str1):
    sequence_num = []
    label_num = []
    for line in open(str1):
        proteinId, sequence, label = line.split(",")
        proteinId = proteinId.strip(' \t\r\n');
        sequence = sequence.strip(' \t\r\n');
        sequence_num.append(trans(sequence))
        label = label.strip(' \t\r\n');
        label_num.append(int(label))

    return sequence_num, label_num


# a, b = createTrainData("positive_and_negative.csv")
# t = (a, b)
# cPickle.dump(t, open("data.pkl", "wb"))


# def createTrainTestData(str_path, nb_words=None, skip_top=0,
#                         maxlen=None, test_split=0.25, seed=113,
#                         start_char=1, oov_char=2, index_from=3):
#     X, labels = cPickle.load(open(str_path, "rb"))
#
#     np.random.seed(seed)
#     np.random.shuffle(X)
#     np.random.seed(seed)
#     np.random.shuffle(labels)
#     if start_char is not None:
#         X = [[start_char] + [w + index_from for w in x] for x in X]
#     elif index_from:
#         X = [[w + index_from for w in x] for x in X]
#
#     if maxlen:
#         new_X = []
#         new_labels = []
#         for x, y in zip(X, labels):
#             if len(x) < maxlen:
#                 new_X.append(x)
#                 new_labels.append(y)
#         X = new_X
#         labels = new_labels
#     if not X:
#         raise Exception('After filtering for sequences shorter than maxlen=' +
#                         str(maxlen) + ', no sequence was kept. '
#                                       'Increase maxlen.')
#     if not nb_words:
#         nb_words = max([max(x) for x in X])
#
#     if oov_char is not None:
#         X = [[oov_char if (w >= nb_words or w < skip_top) else w for w in x] for x in X]
#     else:
#         nX = []
#         for x in X:
#             nx = []
#             for w in x:
#                 if (w >= nb_words or w < skip_top):
#                     nx.append(w)
#             nX.append(nx)
#         X = nX
#
#     X_train = np.array(X[:int(len(X) * (1 - test_split))])
#     y_train = np.array(labels[:int(len(X) * (1 - test_split))])
#
#     X_test = np.array(X[int(len(X) * (1 - test_split)):])
#     y_test = np.array(labels[int(len(X) * (1 - test_split)):])
#
#     return (X_train, y_train), (X_test, y_test)

def precision(y_true, y_pred):
    """Precision metric.

    Only computes a batch-wise average of precision.

    Computes the precision, a metric for multi-label classification of
    how many selected items are relevant.
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


def recall(y_true, y_pred):
    """Recall metric.

    Only computes a batch-wise average of recall.

    Computes the recall, a metric for multi-label classification of
    how many relevant items are selected.
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


def f1(y_true, y_pred):
    def recall(y_true, y_pred):
        """Recall metric.

        Only computes a batch-wise average of recall.

        Computes the recall, a metric for multi-label classification of
        how many relevant items are selected.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        """Precision metric.

        Only computes a batch-wise average of precision.

        Computes the precision, a metric for multi-label classification of
        how many selected items are relevant.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision

    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2 * ((precision * recall) / (precision + recall))


# Embedding
max_features = 23
maxlen = 1000
embedding_size = 128

# Convolution
# filter_length = 3
nb_filter = 512
pool_length = 2

# LSTM
lstm_output_size = 512

# Training

nb_epoch = 100

batch_size = 20
num_classes = 2
num_features = 21000
seq_len = 1000
num_features_per = 21
collection_name = "sensitivity_analysis"
split_size = 5
epochs = 30

CHARSET = {'A': 0, 'C': 1, 'D': 2, 'E': 3, 'F': 4, 'G': 5, 'H': 6, \
           'I': 7, 'K': 8, 'L': 9, 'M': 10, 'N': 11, 'P': 12, 'Q': 13, \
           'R': 14, 'S': 15, 'T': 16, 'V': 17, 'W': 18, 'Y': 19, 'X': 20, \
           'O': 20, 'U': 20,
           'B': (2, 11),
           'Z': (3, 13),
           'J': (7, 9)}
CHARLEN = 21

print('Loading data...')
seed = 123456

negative_dataFile = '/home/myc/projectpy/cnnTensorflowNew/data/CaricaPapaya/train_8000.txt'
positive_dataFile = "/home/myc/projectpy/cnnTensorflowNew/data/CaricaPapaya/CA_group_features_CA_25800.fa.csv"

groupData = get_Data_group1(positive_dataFile)

X_0, Y_0 = get_Data(negative_dataFile)

kfold = StratifiedKFold(n_splits=split_size, shuffle=True, random_state=56465)
current_time = time.time()
time_tag = str(int(current_time))

groupData = groupFilter(groupData, 100)

families = np.array(groupData["family"].tolist())
names = np.array(groupData["mernum"].tolist())
labels = np.array(groupData["label"].tolist())
seqs = groupData["sequence"].tolist()

data = np.zeros((len(seqs), num_features), dtype=np.float32)
for i in range(len(seqs)):
    encoding_seq_np_list(seqs[i], data[i])

kf = GroupKFold(n_splits=split_size)

for (fold, (train_0, test_0)), (train_1, test_1) in zip(enumerate(kfold.split(X_0, Y_0)),
                                                        kf.split(seqs, labels, families)):
    print('\nfold:%s' % fold)
    start_time = datetime.now()

    x_train_0 = X_0[train_0]
    ky_train_0 = Y_0[train_0]

    x_test_0 = X_0[test_0]
    y_test_0 = Y_0[test_0]

    x_train_1 = data[train_1]
    ky_train_1 = labels[train_1]
    z_train_1 = names[train_1]
    family_train_1 = families[train_1]
    print("train family: ", pandas.Series(family_train_1).unique())

    x_test_1 = data[test_1]
    y_test_1 = labels[test_1]
    z_test_1 = names[test_1]
    family_test_1 = families[test_1]
    # print("test family: ", family_test_1)
    x_train = np.vstack((x_train_0, x_train_1))
    y_train = np.append(ky_train_0, ky_train_1, axis=0)

    ros = RandomOverSampler(random_state=6548)
    # ros = SMOTEENN(ratio='minority', n_jobs=6)

    x_train, y_train = ros.fit_sample(x_train, y_train)

    x_train, y_train = shuffle_list(x_train, y_train)

    print("before sample balance treatment:%s, %s " % (len(x_train_0), len(x_train_1)))
    print("after sample balance treatment :%s  " % (len(x_train)))
    print("sample data detail: ", sorted(Counter(y_train).items()))

    x_train = np.reshape(x_train, (len(x_train), 21000, 1))

    x_test = np.vstack((x_test_0, x_test_1))
    x_test = np.reshape(x_test, (len(x_test), 21000, 1))

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')

    y_test = np.append(y_test_0, y_test_1)
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    print('Build model...')

    model = Sequential()
    # model.add(Embedding(max_features, embedding_size, input_length=maxlen))
    # model.add(Dropout(0.5))
    model.add(Convolution1D(filters=nb_filter,
                            kernel_size=9,
                            input_shape=(21000, 1),
                            padding='valid',
                            activation='relu'))
    model.add(MaxPooling1D(pool_size=pool_length))

    model.add(Convolution1D(filters=nb_filter,
                            kernel_size=5,
                            padding='valid',
                            activation='relu'))
    model.add(MaxPooling1D(pool_size=pool_length))

    model.add(LSTM(lstm_output_size))
    model.add(Dense(2))
    model.add(Activation('softmax'))
    model.summary()

    model.compile(loss=keras.losses.categorical_crossentropy,
                  # optimizer=keras.optimizers.Adam(),
                  optimizer='rmsprop',
                  metrics=['accuracy', precision, recall, f1,
                           metrics.categorical_accuracy])

    print('Train...')

    train_data = Dataset(x_train, y_train)

    step = 0

    for epoch in range(epochs):
        total_batch = int(train_data._num_examples / batch_size)
        avg_cost = 0
        avg_acc = 0

        for i in range(total_batch):
            batch_x, batch_y = train_data.next_batch(batch_size)

            history = model.fit(x_train, y_train, batch_size=batch_size, epochs=nb_epoch, validation_split=0.1)

            # counter
            step += 1

    # json_string = model.to_json()
    # open('my_model_rat.json', 'w').write(json_string)
    # model.save_weights('my_model_rat_weights.h5')
    score, acc = model.evaluate(x_test, y_test, batch_size=batch_size)
    print('Test score:', score)
    print('Test accuracy:', acc)
    print('***********************************************************************')
