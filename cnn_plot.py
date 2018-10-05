from __future__ import print_function
import keras
from keras.datasets import mnist
from keras.layers import Dense, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.models import Sequential
from keras import metrics

from sklearn.metrics import classification_report, roc_auc_score, roc_curve, make_scorer, confusion_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_curve, auc
from binaryEval import tp, tn, fp, fn, precision, mcor, precision, recall, f1, plot_confusion_matrix

import numpy
from numpy import argmax
from scipy import interp

import matplotlib
import os
import matplotlib as mpl
if os.environ.get('DISPLAY','') == '':
    print('no display found. Using non-interactive Agg backend')
    mpl.use('Agg')
import matplotlib.pyplot as plt

import tensorflow as tf


config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)

# array length
arrayLength = 259
seed = 123456
numpy.random.seed(seed)

# read file
dataset = numpy.loadtxt("259.csv", delimiter=",")
X = dataset[:, 0:arrayLength]
Y = dataset[:, arrayLength]
batch_size = 500
num_classes = 2
epochs = 50
img_x, img_y = 1, 259

# define 10-fold cross validation test harness
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
cvscores = []
tprs = []
aucs = []
mean_fpr = numpy.linspace(0, 1, 100)
i = 0

for fold, (train, test) in enumerate(kfold.split(X, Y)):
    print('fold:%s' % fold)
    x_train = X[train].reshape(X[train].shape[0], img_x, img_y, 1)
    x_test = X[test].reshape(X[test].shape[0], img_x, img_y, 1)
    input_shape = (img_x, img_y, 1)
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255
    y_train = keras.utils.to_categorical(Y[train], num_classes)
    y_test = keras.utils.to_categorical(Y[test], num_classes)

    model = Sequential()
    model.add(Conv2D(100, kernel_size=(1, 1), strides=(1, 1),
                     activation='relu',
                     input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(1, 1), strides=(1, 1)))
    model.add(Conv2D(200, (1, 1), activation='relu'))
    model.add(MaxPooling2D(pool_size=(1, 1)))
    model.add(Flatten())
    model.add(Dense(1000, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adam(),
                  metrics=['accuracy', precision, mcor, recall, f1, metrics.binary_accuracy, metrics.categorical_accuracy])


    class AccuracyHistory(keras.callbacks.Callback):
        def on_train_begin(self, logs={}): self.acc = []

        def on_epoch_end(self, batch, logs={}): self.acc.append(logs.get('acc'))

    history = AccuracyHistory()

    scoring = {'tp': make_scorer(tp), 'tn': make_scorer(tn), 'fp': make_scorer(fp), 'fn': make_scorer(fn)}

    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              verbose=2,
              validation_data=(x_test, y_test),
              callbacks=[history])

    predict_score = model.predict(x_test, verbose=0)

    targetNames = ['class 0', 'class 1']
    y_test_argmax = argmax(y_test, axis=1)
    predict_argmax = argmax(predict_score, axis=1)

    print(classification_report(y_test_argmax, predict_argmax, target_names=targetNames))

    cnf_matrix = confusion_matrix(y_test_argmax, predict_argmax)

    print(cnf_matrix)

    # print(cnf_matrix.ravel())

    # plt.figure()
    # plot_confusion_matrix(cnf_matrix, classes=targetNames,
    #                       title='Confusion matrix, without normalization')

    print('roc:%.2f%%' % roc_auc_score(y_test_argmax, predict_argmax))

    # Compute ROC curve and area the curve
    fpr, tpr, thresholds = roc_curve(Y[test], predict_score[:, 1])
    tprs.append(interp(mean_fpr, fpr, tpr))
    tprs[-1][0] = 0.0
    roc_auc = auc(fpr, tpr)
    aucs.append(roc_auc)
    plt.plot(fpr, tpr, lw=1, alpha=0.3, label='ROC fold %d (AUC = %0.2f)' % (i, roc_auc))

    i += 1
    scores = model.evaluate(x_test, y_test, verbose=0)
    print("%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))
    cvscores.append(scores[1] * 100)

print("%.2f%% (+/- %.2f%%)" % (numpy.mean(cvscores), numpy.std(cvscores)))


plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='Luck', alpha=.8)

mean_tpr = numpy.mean(tprs, axis=0)
mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr, mean_tpr)
std_auc = numpy.std(aucs)
plt.plot(mean_fpr, mean_tpr, color='b',
         label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
         lw=2, alpha=.8)

std_tpr = numpy.std(tprs, axis=0)
tprs_upper = numpy.minimum(mean_tpr + std_tpr, 1)
tprs_lower = numpy.maximum(mean_tpr - std_tpr, 0)
plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                 label=r'$\pm$ 1 std. dev.')

plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()
plt.savefig('plot.png')