from __future__ import print_function
import sys
import os
import time

import keras
import tensorflow as tf
import tensorflow.contrib.slim as slim

from sklearn.metrics import classification_report, roc_auc_score, roc_curve, make_scorer, confusion_matrix
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from sklearn.metrics import auc, precision_recall_curve

import numpy
from numpy import argmax
from scipy import interp
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

from datetime import datetime

from batchDataset import Dataset

from models_2_4_caricapapaya import MNIST_CNN, Taylor

from imblearn.over_sampling import RandomOverSampler
from imblearn.over_sampling import SMOTE, ADASYN

## output roc plot according to cross-validation,

total_start_time = datetime.now()

batch_size = 50
num_classes = 2
num_features = 21000
seq_len = 1000
num_features_per = 21
collection_name = "sensitivity_analysis"
split_size = 5
epochs = 10

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
    dataFile = "/home/myc/projectpy/cnnTensorflowNew/data/CaricaPapaya/caricapapaya_limit_5_1074.txt"
    print("data File:", dataFile)

    for index, line in enumerate(open(dataFile, 'r').readlines()):
        w = line.split(' ')
        label = w[0]
        features = w[1]
        # if index ==99999:
        #     print("haha")
        #print("data line:", index)

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


def model_summary():
    model_vars = tf.trainable_variables()
    slim.model_analyzer.analyze_vars(model_vars, print_info=True)


img_x, img_y = 1, num_features


def run_test(session, train_accuracy, prediction, x_placeholder, y_placeholder, x_test, y_test, cvscores, tprs, fprs,
             aucs, fold):
    print('testing account:', len(x_test))

    train_data = Dataset(x_test, y_test)
    total_batch = int(train_data._num_examples / batch_size)

    test_accuracy_scores = []
    labels = []
    predict_scores = []
    j = 0
    for i in range(total_batch):
        batch_x, batch_y = train_data.next_batch(batch_size)

        test_accuracy_score = session.run(train_accuracy, feed_dict={x_placeholder: batch_x, y_placeholder: batch_y})

        test_accuracy_scores.append(test_accuracy_score)

        predict_score = session.run(prediction, feed_dict={x_placeholder: batch_x, y_placeholder: batch_y})
        labels.extend(batch_y)
        predict_scores.extend(predict_score)
        j = j + 1

    print('test Accuracy mean:', numpy.mean(test_accuracy_scores))

    targetNames = ['class 0', 'class 1']

    y_test_argmax = argmax(labels, axis=1)
    predict_argmax = argmax(predict_scores, axis=1)

    precisions, recalls, thresholds = precision_recall_curve(argmax(labels, axis=1), argmax(predict_scores, axis=1))
    plot_precision_recall_vs_threshold(precisions, recalls, thresholds, fold)

    # draw precision recall plot and then close plt
    plt.xlabel("Threshold")
    plt.legend(loc='lower right')
    plt.ylim([0, 1])
    plt.savefig('./' + cur_script_name() + '_precision_recall')
    plt.close()

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

    return test_accuracy_score, predict_score


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
    plt.savefig(script_name + ".png")


def cur_script_name():
    argv0_list = sys.argv[0].split("/")
    script_name = argv0_list[len(argv0_list) - 1]  # get script file name self
    print("current script:", script_name)
    script_name = script_name[0:-3]  # remove ".py"

    return script_name


def plot_draw_cross_validation(cvscores, tprso, fprs, aucs):
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
    plt.savefig(script_name + ".png")


def plot_precision_recall_vs_threshold(precisions, recalls, thresholds, fold):
    plt.plot(thresholds, precisions[:-1], label='Precision fold %d' % fold)
    plt.plot(thresholds, recalls[:-1], label='Recall fold %d' % fold)


def cross_validate(X, Y, epochs, class_num, feature_num, collection_name, split_size=5):
    results = []
    hmaps = []
    cvscores = dict()
    tprs = dict()
    fprs = dict()
    aucs = dict()
    checkpoints = []

    seed = 123456

    kfold = StratifiedKFold(n_splits=split_size, shuffle=True, random_state=56465)
    current_time = time.time()
    time_tag = str(int(current_time))

    for fold, (train, test) in enumerate(kfold.split(X, Y)):
        checkpoint = []

        logdir = sys.path[0] + "/log/log_" + time_tag + "_fold_" + str(fold) + "/"
        ckptdir = logdir + '_model'

        if not os.path.exists(ckptdir):
            os.makedirs(ckptdir)

        checkpoint.append(logdir)
        checkpoint.append(ckptdir)

        print('\n fold:%s' % fold)
        start_time = datetime.now()

        x_train = X[train]
        x_train = numpy.reshape(x_train, (len(x_train), 21000))
        ky_train = Y[train]

        x_test = X[test]
        x_test = numpy.reshape(x_test, (len(x_test), 21000))

        print("training data count:", len(x_train))

        x_train = x_train.astype('float32')
        x_test = x_test.astype('float32')

        y_train = keras.utils.to_categorical(ky_train, num_classes)
        y_test = keras.utils.to_categorical(Y[test], num_classes)

        # train
        tf.reset_default_graph()
        with tf.name_scope('Classifier'):

            # Initialize neural network
            DNN = MNIST_CNN('CNN')

            # Setup training process
            x_placeholder = tf.placeholder(tf.float32, [None, 21000], name='x_placeholder')
            y_placeholder = tf.placeholder(tf.float32, [None, 2], name='y_placeholder')
            global_step = tf.placeholder(tf.int32)

            trainingFlag = tf.placeholder_with_default(True, shape=[], name='trainingFlag')

            activations, prediction, logits = DNN(x_placeholder)

            tf.add_to_collection('DTD_T', prediction)
            tf.add_to_collection('DTD_T', logits)
            tf.add_to_collection('DTD_T', y_placeholder)

            tf.add_to_collection('DTD', x_placeholder)

            for activation in activations:
                tf.add_to_collection('DTD', activation)

            tf.add_to_collection('DTD', prediction)

            cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y_placeholder),
                                  name="loss_reduce_mean")

            optimizer = tf.train.AdamOptimizer().minimize(cost, var_list=DNN.vars, name="adam_train_op")

            correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(y_placeholder, 1), name="equal_train_op")
            train_accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name="reduce_mean_op")

            model_summary()

        cost_summary = tf.summary.scalar('Train Cost', cost)
        accuray_summary = tf.summary.scalar('Train Accuracy', train_accuracy)
        summary = tf.summary.merge_all()

        with tf.Session() as sess:

            sess.run(tf.global_variables_initializer())
            print("\n Start training")

            train_data = Dataset(x_train, y_train)

            saver = tf.train.Saver()
            file_writer = tf.summary.FileWriter(logdir, tf.get_default_graph())

            step = 0

            for epoch in range(epochs):
                total_batch = int(train_data._num_examples / batch_size)
                avg_cost = 0
                avg_acc = 0

                for i in range(total_batch):
                    batch_x, batch_y = train_data.next_batch(batch_size)

                    _, c, a, summary_str = sess.run([optimizer, cost, train_accuracy, summary],
                                                    feed_dict={x_placeholder: batch_x,
                                                               y_placeholder: batch_y, global_step: step},
                                                    )
                    avg_cost += c / total_batch
                    avg_acc += a / total_batch

                    file_writer.add_summary(summary_str, epoch * total_batch + i)
                    # counter
                    step += 1

                print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.9f}'.format(avg_cost), 'accuracy =',
                      '{:.9f}'.format(avg_acc))

            saver.save(sess, ckptdir)
                # print('Accuracy:', session.run(accuracy, feed_dict={X: mnist.test.images, Y: mnist.test.labels}))

                # print('Accuracy:', session.run(train_accuracy, feed_dict={x_placeholder: x_test, y_placeholder: y_test}))
                # test
            test_accuracy_score, predict_score = run_test(sess, train_accuracy, prediction, x_placeholder,
                                                              y_placeholder, x_test, y_test, cvscores, tprs, fprs, aucs,
                                                              fold)
            results.append(test_accuracy_score)

        end_time = datetime.now()

        print("The ", fold, " fold Duration: {}".format(end_time - start_time))

    # draw plot
    plot_draw_cross_validation(cvscores, tprs, fprs, aucs)

    return results, checkpoints


hmaps = []

X, Y = get_Data()

ros = RandomOverSampler(random_state=0)

X_resampled, y_resampled = ros.fit_sample(X, Y)

from collections import Counter
print(sorted(Counter(y_resampled).items()))

print("before sample balance treatment:", sorted(Counter(Y).items()))
print("after sample balance treatment :", sorted(Counter(y_resampled).items()))


results, checkpoints = cross_validate(X_resampled, y_resampled, epochs, num_classes, num_features,
                                      collection_name, split_size)

# print("Cross-validation test accuracy: %s" % results)
# print("Test accuracy: %f" % session.run(accuracy, feed_dict={x_placeholder: test_x, y_placeholder: test_y}))

# hmaps = run_deep_taylor_decomposition(checkpoints, num_classes, num_features, collection_name)
#
# nhmaps = numpy.array(hmaps)
# numpy.save(cur_script_name() + "hmaps", nhmaps)
#
# print("relevance scores: %s" % hmaps)

total_end_time = datetime.now()
print("/n/n total duration : {}".format(total_end_time, total_start_time))
