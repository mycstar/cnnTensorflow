from __future__ import print_function
import sys, os
from random import randint

import keras
from keras import backend as K
import tensorflow as tf
from keras.losses import binary_crossentropy

from sklearn.metrics import classification_report, roc_auc_score, roc_curve, make_scorer, confusion_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_curve, auc

import numpy
from numpy import argmax
from scipy import interp
import pandas
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

import functools
from itertools import product
import csv
from datetime import datetime

from batchDataset import Dataset


def tp(y_true, y_pred): return confusion_matrix(y_true, y_pred)[0, 0]


def tn(y_true, y_pred): return confusion_matrix(y_true, y_pred)[0, 0]


def fp(y_true, y_pred): return confusion_matrix(y_true, y_pred)[1, 0]


def fn(y_true, y_pred): return confusion_matrix(y_true, y_pred)[0, 1]


def precision(y_true, y_pred):
    # Count positive samples.
    c1 = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    c2 = K.sum(K.round(K.clip(y_pred, 0, 1)))
    c3 = K.sum(K.round(K.clip(y_true, 0, 1)))

    # If there are no true samples, fix the F1 score at 0.
    if c3 == 0:
        return 0

    # How many selected items are relevant?
    precision = c1 / c2

    return precision


def mcor(y_true, y_pred):
    # matthews_correlation
    y_pred_pos = K.round(K.clip(y_pred, 0, 1))
    y_pred_neg = 1 - y_pred_pos

    y_pos = K.round(K.clip(y_true, 0, 1))
    y_neg = 1 - y_pos

    tp = K.sum(y_pos * y_pred_pos)
    tn = K.sum(y_neg * y_pred_neg)

    fp = K.sum(y_neg * y_pred_pos)
    fn = K.sum(y_pos * y_pred_neg)

    numerator = (tp * tn - fp * fn)
    denominator = K.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))

    return numerator / (denominator + K.epsilon())


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


def generate_batch_data_random(x, y, batch_size):
    ylen = len(y)
    loopcount = ylen // batch_size
    while (True):
        i = randint(0, loopcount)
        yield x[i * batch_size:(i + 1) * batch_size], y[i * batch_size:(i + 1) * batch_size]


def w_categorical_crossentropy(y_true, y_pred, weights):
    nb_cl = len(weights)
    final_mask = K.zeros_like(y_pred[:, 0])
    y_pred_max = K.max(y_pred, axis=1)
    y_pred_max = K.reshape(y_pred_max, (K.shape(y_pred)[0], 1))
    y_pred_max_mat = K.cast(K.equal(y_pred, y_pred_max), K.floatx())
    for c_p, c_t in product(range(nb_cl), range(nb_cl)):
        final_mask += (weights[c_t, c_p] * y_pred_max_mat[:, c_p] * y_true[:, c_t])
    return K.categorical_crossentropy(y_pred, y_true) * final_mask


def custom_loss_4(y_true, y_pred, weights):
    return K.mean(K.abs(y_true - y_pred) * weights)


def custom_loss(real_value):
    def loss(y_true, y_pred):
        if y_true == 1:
            return binary_crossentropy(y_true, y_pred) * 1000
        else:
            return binary_crossentropy(y_true, y_pred)

    return loss


# bp mll loss function
# y_true, y_pred must be 2D tensors of shape (batch dimension, number of labels)
# y_true must satisfy y_true[i][j] == 1 iff sample i has label j
# compute pairwise differences between elements of the tensors a and b
def pairwise_sub(a, b):
    column = K.expand_dims(a, 2)
    row = K.expand_dims(b, 1)
    return column - row


# compute pairwise logical and between elements of the tensors a and b
def pairwise_and(a, b):
    column = K.expand_dims(a, 2)
    row = K.expand_dims(b, 1)
    return K.minimum(column, row)


def bp_mll_loss(y_true, y_pred):
    # get true and false labels
    y_i = K.equal(y_true, K.ones_like(y_true))
    y_i_bar = K.not_equal(y_true, K.ones_like(y_true))

    # cast to float as keras backend has no logical and
    y_i = K.cast(y_i, dtype='float32')
    y_i_bar = K.cast(y_i_bar, dtype='float32')

    # get indices to check
    truth_matrix = pairwise_and(y_i, y_i_bar)

    # calculate all exp'd differences
    sub_matrix = pairwise_sub(y_pred, y_pred)
    exp_matrix = K.exp(-sub_matrix)

    # check which differences to consider and sum them
    sparse_matrix = exp_matrix * truth_matrix
    sums = K.sum(sparse_matrix, axis=[1, 2])

    # get normalizing terms and apply them
    y_i_sizes = K.sum(y_i, axis=1)
    y_i_bar_sizes = K.sum(y_i_bar, axis=1)
    normalizers = y_i_sizes * y_i_bar_sizes
    results = sums / normalizers

    # sum over samples
    return K.sum(results)


def ptb_iterator(raw_data, batch_size, num_steps):
    """Iterate on the raw PTB data.

    This generates batch_size pointers into the raw PTB data, and allows
    minibatch iteration along these pointers.

    Args:
      raw_data: one of the raw data outputs from ptb_raw_data.
      batch_size: int, the batch size.
      num_steps: int, the number of unrolls.

    Yields:
      Pairs of the batched data, each a matrix of shape [batch_size, num_steps].
      The second element of the tuple is the same data time-shifted to the
      right by one.

    Raises:
      ValueError: if batch_size or num_steps are too high.
    """
    raw_data = numpy.array(raw_data, dtype=numpy.int32)

    data_len = len(raw_data)
    batch_len = data_len // batch_size  # 有多少个batch
    data = numpy.zeros([batch_size, batch_len], dtype=numpy.int32)  # batch_len 有多少个单词
    for i in range(batch_size):  # batch_size 有多少个batch
        data[i] = raw_data[batch_len * i:batch_len * (i + 1)]

    epoch_size = (batch_len - 1) // num_steps  # batch_len 是指一个batch中有多少个句子
    # epoch_size = ((len(data) // model.batch_size) - 1) // model.num_steps  # // 表示整数除法
    if epoch_size == 0:
        raise ValueError("epoch_size == 0, decrease batch_size or num_steps")

    for i in range(epoch_size):
        x = data[:, i * num_steps:(i + 1) * num_steps]
        y = data[:, i * num_steps + 1:(i + 1) * num_steps + 1]
        yield (x, y)


def next_batch(train_data, train_target, batch_size):
    index = [i for i in range(0, len(train_target))]
    numpy.random.shuffle(index)
    batch_data = []
    batch_target = []
    for i in range(0, batch_size):
        batch_data.append(train_data[index[i]])
        batch_target.append(train_target[index[i]])
    return batch_data, batch_target


class AccuracyHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}): self.acc = []

    def on_epoch_end(self, batch, logs={}): self.acc.append(logs.get('acc'))


X = []
Y = []
allArray = []

total_start_time = datetime.now()

dataFile = "../data/259_2000.csv"
print("data File:", dataFile)

for index, line in enumerate(open(dataFile, 'r').readlines()):
    w = line.split(',')
    label = w[-1:]
    features = w[:-1]

    try:
        label = [int(x) for x in label]
        features = [float(x) for x in features]
    except ValueError:
        print('Line %s is corrupt!' % index)
        break

    X.append(features)
    Y.extend(label)

X = numpy.asarray(X)
Y = numpy.asarray(Y)
# Y = Y.reshape(Y.shape[0], 1)

print("feature shape:", X.shape)
print("label shape:", Y.shape)

# array length
arrayLength = X.shape[1]
seed = 123456
numpy.random.seed(seed)

batch_size = 50
num_classes = 2
epochs = 2
img_x, img_y = 1, X.shape[1]

# define 10-fold cross validation test harness
kfold = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
cvscores = []
tprs = []
aucs = []
mean_fpr = numpy.linspace(0, 1, 100)
i = 0

for fold, (train, test) in enumerate(kfold.split(X, Y)):

    logdir = "./log_taylor_2_fold_" + str(fold) + "/"
    ckptdir = logdir + '_model'

    if not os.path.exists(logdir):
        os.mkdir(logdir)

    if not os.path.exists(ckptdir):
        os.mkdir(ckptdir)

    print('fold:%s' % fold)
    start_time = datetime.now()

    input_shape = (img_x, img_y, 1)

    kx_train = X[train]
    ky_train = Y[train]
    kx_train = kx_train.astype('float32')
    print("training data count before Smoteenn:", len(kx_train))

    kx_train_res, ky_train_res = kx_train, ky_train

    print("training data count after Smoteenn:", len(kx_train_res))

    # x_train = kx_train_res.reshape(kx_train_res.shape[0], img_x, img_y, 1)
    x_train = kx_train_res
    x_test = X[test]

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')

    y_train = keras.utils.to_categorical(ky_train_res, num_classes)
    y_test = keras.utils.to_categorical(Y[test], num_classes)

    train_data = Dataset(x_train, y_train)

    x_placeholder = tf.placeholder(tf.float32, [None, img_y])
    y_placeholder = tf.placeholder(tf.float32, [None, 2], name="truth")

    trainingFlag = tf.placeholder_with_default(True, shape=[], name='training')

    reshaped_input = tf.reshape(x_placeholder, [-1, img_x, img_y, 1], name="absolute_input")

    y_pred = tf.keras.layers.Conv2D(filters=128, kernel_size=[3, 3], strides=(1, 1), padding='same', activation='relu')(
        reshaped_input)
    y_pred = tf.keras.layers.MaxPool2D(pool_size=(1, 1), strides=(1, 1))(y_pred)
    y_pred = tf.keras.layers.Flatten()(y_pred)
    y_pred = tf.keras.layers.Dense(256, activation='relu')(y_pred)

    y_pred = tf.keras.layers.Dense(2, activation='softmax', name="absolute_output")(y_pred)

    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_pred, labels=y_placeholder))

    train_optim = tf.train.AdamOptimizer().minimize(loss)

    correct_prediction = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y_placeholder, 1))
    train_accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    tf.add_to_collection('sensitivity_analysis', trainingFlag)
    tf.add_to_collection('sensitivity_analysis', x_placeholder)
    tf.add_to_collection('sensitivity_analysis', y_pred)

    cost_summary = tf.summary.scalar('Train Cost', loss)
    accuray_summary = tf.summary.scalar('Train Accuracy', train_accuracy)
    summary = tf.summary.merge_all()

    saver = tf.train.Saver()
    file_writer = tf.summary.FileWriter(logdir, tf.get_default_graph())

    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)

        for epoch in range(epochs):
            total_batch = int(train_data._num_examples / batch_size)
            avg_cost = 0
            avg_acc = 0

            for _ in range(10):
                batch_x, batch_y = train_data.next_batch(batch_size)
                _, c, a, summary_str = sess.run([train_optim, loss, train_accuracy, summary],
                                                feed_dict={x_placeholder: batch_x, y_placeholder: batch_y})
                avg_cost += c / total_batch
                avg_acc += a / total_batch

                file_writer.add_summary(summary_str, epoch * total_batch + i)

            print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.9f}'.format(avg_cost), 'accuracy =',
                  '{:.9f}'.format(avg_acc))

            saver.save(sess, ckptdir)

        test_accuracy_score = sess.run(train_accuracy, feed_dict={x_placeholder: x_test, y_placeholder: y_test})
        print('test Accuracy:', test_accuracy_score)

        predict_score = sess.run(y_pred, feed_dict={x_placeholder: x_test, y_placeholder: y_test})

        scoring = {'tp': make_scorer(tp), 'tn': make_scorer(tn), 'fp': make_scorer(fp), 'fn': make_scorer(fn)}

        targetNames = ['class 0', 'class 1']
        y_test_argmax = argmax(y_test, axis=1)
        predict_argmax = argmax(predict_score, axis=1)

        print(classification_report(y_test_argmax, predict_argmax, target_names=targetNames))

        cnf_matrix = confusion_matrix(y_test_argmax, predict_argmax)

        print(cnf_matrix)

        print('roc:%.2f%%' % roc_auc_score(y_test_argmax, predict_argmax))

        # Compute ROC curve and area the curve
        fpr, tpr, thresholds = roc_curve(Y[test], predict_score[:, 1])
        # fpr, tpr, thresholds = roc_curve(y_test_argmax, predict_argmax)

        tprs.append(interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)
        plt.plot(fpr, tpr, lw=1, alpha=0.3, label='ROC fold %d (AUC = %0.2f)' % (i, roc_auc))

        i += 1
        scores = predict_score
        cvscores.append(scores[1] * 100)

    tf.reset_default_graph()

    sess = tf.InteractiveSession()

    new_saver = tf.train.import_meta_graph(ckptdir + '.meta')
    new_saver.restore(sess, tf.train.latest_checkpoint(logdir))

    SA = tf.get_collection('sensitivity_analysis')
    training = SA[0]
    new_x_placeholder = SA[1]
    new_y_pred = SA[2]

    SA_scores = [tf.square(tf.gradients(new_y_pred[:, i], new_x_placeholder)) for i in range(2)]

    images = x_train
    labels = y_train

    sample_imgs = []
    for i in range(2):
        sample_imgs.append(images[numpy.argmax(labels, axis=1) == i][3])

    hmaps = numpy.reshape(
        [sess.run(SA_scores[i], feed_dict={new_x_placeholder: sample_imgs[i][None, :], training: False}) for i in
         range(2)],
        [2, 259])

    sess.close()

    end_time = datetime.now()

    print("The ", fold, " fold Duration: {}".format(end_time - start_time))

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
# plt.show()
argv0_list = sys.argv[0].split("/")
script_name = argv0_list[len(argv0_list) - 1]  # get script file name self
print("current script:", script_name)
script_name = script_name[0:-3]  # remove ".py"
script_num = script_name.split('_')[2]
plt.savefig(script_name + ".png")

total_end_time = datetime.now()
print(" total duration : {}".format(total_end_time, total_start_time))
