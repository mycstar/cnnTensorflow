from __future__ import print_function
import sys
import os
import time

import keras
import tensorflow as tf
import tensorflow.contrib.slim as slim
import pandas

from sklearn.metrics import classification_report, roc_auc_score, roc_curve, make_scorer, confusion_matrix
from sklearn.model_selection import StratifiedKFold, GroupKFold
from sklearn.metrics import auc, precision_recall_curve

import numpy
from numpy import argmax
from scipy import interp
from random import shuffle
from collections import Counter

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

from datetime import datetime

from taylorDecomposition.batchDataset import Dataset

from deepFam_model_caricapapaya import get_placeholders, inference

from imblearn.combine import SMOTEENN

## output roc plot according to cross-validation,

total_start_time = datetime.now()

batch_size = 10
num_classes = 2
num_features = 21000
num_features_per = 21
seq_len = 1000
collection_name = "sensitivity_analysis"
split_size = 5
epochs = 30
num_windows = [256, 256, 256, 256, 256, 256, 256, 256, 256]
window_lengths = [8, 12, 16, 20, 24, 28, 32, 36, 40]
num_hidden = 512
keep_prob = 0.7
regularizer = 0.001
learning_rate = 0.005
img_x, img_y = 1, num_features

CHARSET = {'A': 0, 'C': 1, 'D': 2, 'E': 3, 'F': 4, 'G': 5, 'H': 6, \
           'I': 7, 'K': 8, 'L': 9, 'M': 10, 'N': 11, 'P': 12, 'Q': 13, \
           'R': 14, 'S': 15, 'T': 16, 'V': 17, 'W': 18, 'Y': 19, 'X': 20, \
           'O': 20, 'U': 20,
           'B': (2, 11),
           'Z': (3, 13),
           'J': (7, 9)}
CHARLEN = 21


def encoding_seq_np(seq):
    arr = numpy.zeros((1, CHARLEN * 1000), dtype=numpy.float32)
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


def get_Data():
    X = []
    Y = []
    dataFile = "/home/myc/projectpy/cnnTensorflowNew/data/CaricaPapaya/caricapapaya_label_0_500.txt"
    print("data File:", dataFile)

    for index, line in enumerate(open(dataFile, 'r').readlines()):
        w = line.split(' ')
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

    X = numpy.asarray(X)
    X = numpy.reshape(X, (len(X), 21000))

    Y = numpy.asarray(Y)

    print("feature shape:", X.shape)
    print("label shape:", Y.shape)

    return X, Y


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

def get_Data_group1():
    dataFile = "/home/myc/projectpy/cnnTensorflowNew/data/CaricaPapaya/CA_group_features_CA_25800.fa.csv"
    print("data File:", dataFile)

    groupData = pandas.read_csv(dataFile, sep=",", header=0)
    groupData["sequence"] = groupData["sequence"].astype(str)
    print("total raw row :", len(groupData))
    groupData = groupData[(groupData['sequence'].str.len() > 50) & (groupData['sequence'].str.len() < 1000)]
    print("after remove invalid sequence <50 and > 1000, left row :", len(groupData))
    groupData['label'] = 1

    return groupData


def groupFilter(df, num):
    groups = pandas.Series(df["family"].tolist()).unique()
    print("total different group is:", len(groups))
    group_series = df.groupby(['family'])['label'].sum()
    ret_family_list = group_series[group_series >= num].keys().tolist()
    print("%s group memebers number are greater than %s" % (len(ret_family_list), num))

    ret_df = df.loc[df['family'].isin(ret_family_list)]
    ret_df = ret_df.groupby(['family']).head(num)
    print("%s samples per group, total samples is:%s" % (num, len(ret_df)))

    return ret_df


def model_summary():
    model_vars = tf.trainable_variables()
    slim.model_analyzer.analyze_vars(model_vars, print_info=True)


def shuffle_list(*ls):
    l = list(zip(*ls))

    shuffle(l)
    return zip(*l)


def randomize(a, b):
    # Generate the permutation index array.
    permutation = numpy.random.permutation(a.shape[0])
    # Shuffle the arrays by giving the permutation in the square brackets.
    shuffled_a = a[permutation]
    shuffled_b = b[permutation]
    return shuffled_a, shuffled_b


def run_test(checkpoint_path, x_test, y_test, cvscores, tprs, fprs, aucs, fold):
    print('testing account:', len(x_test))

    with tf.Graph().as_default():
        # placeholder
        placeholders = get_placeholders(num_features, num_classes)

        # prediction
        pred, layers = inference(placeholders['data'], seq_len, num_features_per, num_classes, window_lengths,
                                 num_windows,
                                 num_hidden, keep_prob, regularizer,
                                 for_training=False)

        prediction = tf.nn.softmax(pred)

        # calculate prediction
        # accuracy
        _acc_op = tf.equal(tf.argmax(pred, 1), tf.argmax(placeholders['labels'], 1))
        train_accuracy = tf.reduce_mean(tf.cast(_acc_op, tf.float32))

        # create saver
        saver = tf.train.Saver()

        # summary
        summary_op = tf.summary.merge_all()

        with tf.Session() as sess:
            # load model
            ckpt = tf.train.latest_checkpoint(os.path.dirname(checkpoint_path))
            if tf.train.checkpoint_exists(ckpt):
                saver.restore(sess, ckpt)
                global_step = ckpt.split('/')[-1].split('-')[-1]

            else:
                # logging("[ERROR] Checkpoint not exist", FLAGS)
                raise Exception('[ERROR] Checkpoint not exist: {}'.checkpoint_path)
                return

            dataset = Dataset(x_test, y_test)

            total_batch = int(dataset._num_examples / batch_size)

            test_accuracy_scores = []
            predict_scores = []
            labelss = []
            num_sample = 0

            for i in range(total_batch):
                data, labels = dataset.next_batch(batch_size, shuffle=False)

                predict_score, test_accuracy_score = sess.run([prediction, train_accuracy],
                                                              feed_dict={placeholders['data']: data,
                                                                         placeholders['labels']: labels})

                print('test Accuracy:', test_accuracy_score)
                test_accuracy_scores.append(test_accuracy_score)

                predict_scores.extend(predict_score)
                labelss.extend(labels)
                num_sample += len(data)

    # labelss = numpy.reshape(labelss, (num_sample, num_classes))
    # predict_scores = numpy.reshape(predict_scores, (num_sample, num_classes))

    targetNames = ['class 0', 'class 1']
    y_test_argmax = argmax(labelss, axis=1)
    predict_argmax = argmax(predict_scores, axis=1)

    print(classification_report(y_true=y_test_argmax, y_pred=predict_argmax, target_names=targetNames))

    cnf_matrix = confusion_matrix(y_test_argmax, predict_argmax)

    print(cnf_matrix)

    print('roc:%.2f%%' % roc_auc_score(y_test_argmax, predict_argmax))

    # Compute ROC curve and area the curve
    fpr, tpr, thresholds = roc_curve(y_test_argmax, predict_argmax)

    mean_fpr = numpy.linspace(0, 1, 100)

    fprs[fold] = fpr
    # print(str(fold), ' fold fpr:', str(len(fpr)))
    tprs[fold] = tpr
    # print(str(fold), ' fold tpr:', str(len(tpr)))

    roc_auc = auc(fpr, tpr)
    aucs[fold] = roc_auc

    scores = predict_score
    cvscores[fold] = (scores[1] * 100)

    return test_accuracy_score, labelss, predict_scores


def plot_draw(cvscores, tprs, fprs, aucs):
    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='navy', label='Luck', alpha=.8)

    # mean_tpr = numpy.mean(tprs, axis=0)
    # mean_tpr[-1] = 1.0
    # mean_auc = auc(mean_fpr, mean_tpr)
    # std_auc = numpy.std(aucs)
    for i in range(len(fprs)):
        plt.plot(fprs[i], tprs[i], label=r'Mean ROC (AUC = %0.2f)' % aucs[i], lw=2, alpha=.8)

    # std_tpr = numpy.std(tprs, axis=0)
    # tprs_upper = numpy.minimum(mean_tpr + std_tpr, 1)
    # tprs_lower = numpy.maximum(mean_tpr - std_tpr, 0)
    # plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
    #                  label=r'$\pm$ 1 std. dev.')

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
    plt.savefig('./return/' + script_name + ".png")


def cur_script_name():
    argv0_list = sys.argv[0].split("/")
    script_name = argv0_list[len(argv0_list) - 1]  # get script file name self
    print("current script:", script_name)
    script_name = script_name[0:-3]  # remove ".py"
    return script_name


def plot_draw_cross_validation(fold, cvscores, tprso, fprs, aucs):
    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='navy', label='Luck', alpha=.8)
    tprs = []
    mean_fpr = numpy.linspace(0, 1, 100)

    for i in range(len(tprso)):
        tprs.append(interp(mean_fpr, fprs[i], tprso[i]))

    for i in range(len(fprs)):
        plt.plot(fprs[i], tprso[i], alpha=0.3, label='ROC fold %d (AUC = %0.2f)' % (i, aucs[i]))
        # plt.plot(fprs[i], tprs[i], label=r'Mean ROC (AUC = %0.2f)' % aucs[i], lw=2, alpha=.8)

    mean_tpr = numpy.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = numpy.std(list(aucs.values()))
    plt.plot(mean_fpr, mean_tpr, label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc), alpha=.8)

    std_tpr = numpy.std(tprs, axis=0)
    tprs_upper = numpy.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = numpy.maximum(mean_tpr - std_tpr, 0)
    plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2, label=r'$\pm$ 1 std. dev.')

    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    # plt.show()
    script_name = cur_script_name()
    plt.savefig('./return/' + script_name + "_fold" + str(fold) + ".png")


def plot_precision_recall_vs_threshold(precisions, recalls, thresholds, fold):
    plt.plot(thresholds, precisions[:-1], label='Precision fold %d' % fold)
    plt.plot(thresholds, recalls[:-1], label='Recall fold %d' % fold)


def cross_validate(X, Y, groupData):
    results = []
    hmaps = []
    cvscores = dict()
    tprs = dict()
    fprs = dict()
    aucs = dict()
    checkpoints = []

    seed = 123456

    groupData = groupFilter(groupData, 30)

    families = numpy.array(groupData["family"].tolist())
    names = numpy.array(groupData["mernum"].tolist())
    labels = numpy.array(groupData["label"].tolist())
    seqs = groupData["sequence"].tolist()

    data = numpy.zeros((len(seqs), num_features), dtype=numpy.float32)
    for i in range(len(seqs)):
        encoding_seq_np_list(seqs[i], data[i])

    kf = GroupKFold(n_splits=split_size)

    kfold = StratifiedKFold(n_splits=split_size, shuffle=True, random_state=5)
    current_time = time.time()
    time_tag = str(int(current_time))

    for (fold, (train_0, test_0)), (train_1, test_1) in zip(enumerate(kfold.split(X, Y)),
                                                        kf.split(seqs, labels, families)):
        checkpoint = []
        print('\nfold:%s' % fold)
        start_time = datetime.now()

        logdir = sys.path[0] + "/log/log_" + time_tag + "_fold_" + str(fold) + "/"
        ckptdir = logdir + '_model'

        if not os.path.exists(ckptdir):
            os.makedirs(ckptdir)

        checkpoint.append(logdir)
        checkpoint.append(ckptdir)

        x_train_0 = X[train_0]
        ky_train_0 = Y[train_0]
        x_test_0 = X[test_0]
        y_test_0 = Y[test_0]

        x_train_1 = data[train_1]
        ky_train_1 = labels[train_1]
        z_train_1 = names[train_1]
        family_train_1 = families[train_1]

        x_test_1 = data[test_1]
        y_test_1 = labels[test_1]
        z_test_1 = names[test_1]
        family_test_1 = families[test_1]

        x_train = numpy.vstack((x_train_0, x_train_1))
        ky_train = numpy.append(ky_train_0, ky_train_1, axis=0)

        # sm = SMOTEENN(ratio='minority', n_jobs=6)
        # x_train, ky_train = sm.fit_sample(x_train, numpy.ravel(ky_train))
        #
        # x_test = X[test]

        print("before sample balance treatment:%s, %s " % (len(x_train_0), len(x_train_1)))
        print("after sample balance treatment :%s  " % (len(x_train)))
        print("sample data detail: ", sorted(Counter(ky_train).items()))

        # ros = RandomOverSampler(random_state=6548)
        # ros = SMOTEENN(ratio='auto', n_jobs=6)
        #
        # x_train, ky_train = ros.fit_sample(x_train, ky_train)

        x_train, ky_train = randomize(x_train, ky_train)

        x_train = numpy.reshape(x_train, (len(x_train), 21000))

        x_test = numpy.vstack((x_test_0, x_test_1))
        x_test = numpy.reshape(x_test, (len(x_test), 21000))
        y_test = numpy.append(y_test_0, y_test_1)

        x_train = x_train.astype('float32')
        x_test = x_test.astype('float32')

        y_train = keras.utils.to_categorical(ky_train, num_classes)
        y_test = keras.utils.to_categorical(y_test, num_classes)
        # train

        with tf.Graph().as_default():
            # get placeholders
            global_step = tf.placeholder(tf.int32)
            placeholders = get_placeholders(num_features, num_classes)

            # prediction
            pred, layers = inference(placeholders['data'], seq_len, num_features_per, num_classes, window_lengths,
                                     num_windows,
                                     num_hidden, keep_prob, regularizer,
                                     for_training=True)

            tf.losses.softmax_cross_entropy(placeholders['labels'], pred)
            cost = tf.losses.get_total_loss()

            # accuracy
            _acc_op = tf.equal(tf.argmax(pred, 1), tf.argmax(placeholders['labels'], 1))
            train_accuracy = tf.reduce_mean(tf.cast(_acc_op, tf.float32))

            # optimization
            optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

            cost_summary = tf.summary.scalar('Train loss', cost)
            accuray_summary = tf.summary.scalar('Train acc_op', train_accuracy)
            summary = tf.summary.merge_all()

            model_summary()

            with tf.Session() as sess:
                sess.run(tf.global_variables_initializer())
                print("\nStart training")

                train_data = Dataset(x_train, y_train)

                saver = tf.train.Saver()
                file_writer = tf.summary.FileWriter(logdir, tf.get_default_graph())

                for epoch in range(epochs):
                    total_batch = int(train_data._num_examples / batch_size)
                    avg_cost = 0
                    avg_acc = 0

                    for i in range(total_batch):
                        batch_x, batch_y = train_data.next_batch(batch_size)
                        _, c, a, summary_str = sess.run([optimizer, cost, train_accuracy, summary],
                                                        feed_dict={placeholders['data']: batch_x,
                                                                   placeholders['labels']: batch_y})
                        avg_cost += c / total_batch
                        avg_acc += a / total_batch

                        file_writer.add_summary(summary_str, epoch * total_batch + i)

                    print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.9f}'.format(avg_cost), 'accuracy =',
                          '{:.9f}'.format(avg_acc))

                    saver.save(sess, ckptdir)
                # print('Accuracy:', session.run(accuracy, feed_dict={X: mnist.test.images, Y: mnist.test.labels}))

            # print('Accuracy:', session.run(train_accuracy, feed_dict={x_placeholder: x_test, y_placeholder: y_test}))
            # test

        test_accuracy_score, lables, predict_score = run_test(ckptdir, x_test, y_test,
                                                              cvscores, tprs, fprs, aucs, fold)
        results.append(test_accuracy_score)

        precisions, recalls, thresholds = precision_recall_curve(argmax(lables, axis=1), argmax(predict_score, axis=1))
        plot_precision_recall_vs_threshold(precisions, recalls, thresholds, fold)

        end_time = datetime.now()

        print("The ", fold, " fold Duration: {}".format(end_time - start_time))

        # draw precision recall plot and then close plt
        plt.xlabel("Threshold")
        plt.legend(loc='lower right')
        plt.ylim([0, 1])
        plt.savefig('./return/' + cur_script_name() + "_fold" + str(fold) + '_precision_recall')
        plt.close()

        # draw plot
        plot_draw_cross_validation(fold, cvscores, tprs, fprs, aucs)


X, Y = get_Data()
groupData = get_Data_group1()

cross_validate(X, Y, groupData)

# print("Cross-validation test accuracy: %s" % results)
# print("Test accuracy: %f" % session.run(accuracy, feed_dict={x_placeholder: test_x, y_placeholder: test_y}))

total_end_time = datetime.now()
print("/n/n total duration : {}".format(total_end_time, total_start_time))
