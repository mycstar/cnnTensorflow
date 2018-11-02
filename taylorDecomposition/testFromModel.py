from __future__ import print_function
import sys
import os
import time

import keras
import tensorflow as tf
import tensorflow.contrib.slim as slim

from sklearn.metrics import classification_report, roc_auc_score, roc_curve, make_scorer, confusion_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import auc, precision_recall_curve

import numpy
from numpy import argmax
from scipy import interp
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

from itertools import cycle

from datetime import datetime

from dataset import DataSet

## output roc plot according to cross-validation,

total_start_time = datetime.now()

batch_size = 100
num_classes = 1074
num_features = 21000

img_x, img_y = 1, num_features


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


def plot_specific_class(fpr, tpr, roc_auc, class_num):
    plt.figure()
    lw = 2
    plt.plot(fpr[class_num], tpr[class_num], color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % roc_auc[class_num])
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()


def plot_multi_class(fpr, tpr, roc_auc):
    # Compute macro-average ROC curve and ROC area
    lw = 2
    # First aggregate all false positive rates
    all_fpr = numpy.unique(numpy.concatenate([fpr[i] for i in range(num_classes)]))

    # Then interpolate all ROC curves at this points
    mean_tpr = numpy.zeros_like(all_fpr)
    for i in range(num_classes):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
    mean_tpr /= num_classes

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    # Plot all ROC curves
    plt.figure()
    plt.plot(fpr["micro"], tpr["micro"],
             label='micro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["micro"]),
             color='deeppink', linestyle=':', linewidth=4)

    plt.plot(fpr["macro"], tpr["macro"],
             label='macro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["macro"]),
             color='navy', linestyle=':', linewidth=4)

    colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
    for i, color in zip(range(num_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                 label='ROC curve of class {0} (area = {1:0.2f})'
                       ''.format(i, roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Some extension of Receiver operating characteristic to multi-class')
    plt.legend(loc="lower right")
    script_name = cur_script_name()
    plt.savefig(script_name + "_multi.png")


hmaps = []

with tf.Session() as session:
    # test_file = "/home/myc/projectpy/DeepFam/data/COG-500-1074/90percent/test.txt"
    # test_file = "/home/myc/projectpy/cnnTensorflowNew/taylorDecomposition/test70.txt"
    test_file = "/home/myc/projectpy/DeepFam/data/ps51371/data1.txt"
    dataset = DataSet(fpath=test_file,
                      seqlen=1000,
                      n_classes=1074,
                      need_shuffle=False)

    test_accuracy_scores = []
    fold = 0

    # ckptdir = "/home/myc/projectpy/cnnTensorflowNew/taylorDecomposition/log/log_1540792513_fold_0/_model"
    # logdir = "/home/myc/projectpy/cnnTensorflowNew/taylorDecomposition/log/log_1540792513_fold_0"

    # ckptdir = "/home/myc/projectpy/cnnTensorflowNew/taylorDecomposition/log/log_1540842612_fold_0/_model"
    # logdir = "/home/myc/projectpy/cnnTensorflowNew/taylorDecomposition/log/log_1540842612_fold_0"
    # ckptdir = "/home/myc/projectpy/cnnTensorflowNew/taylorDecomposition/log/log_1540964590_fold_0/_model"
    # logdir = "/home/myc/projectpy/cnnTensorflowNew/taylorDecomposition/log/log_1540964590_fold_0"

    ckptdir = "/home/myc/projectpy/cnnTensorflowNew/taylorDecomposition/log/log_1541007951_fold_0/_model"
    logdir = "/home/myc/projectpy/cnnTensorflowNew/taylorDecomposition/log/log_1541007951_fold_0"
    # init_op = tf.initialize_all_variables()

    # sess.run(init_op)
    # x_placeholder = tf.placeholder(tf.float32, [None, 21000], name='x_placeholder')
    # y_placeholder = tf.placeholder(tf.float32, [None, 1074], name='y_placeholder')

    # trainingFlag = tf.placeholder_with_default(True, shape=[], name='trainingFlag')

    new_saver = tf.train.import_meta_graph(ckptdir + '.meta')
    new_saver.restore(session, tf.train.latest_checkpoint(logdir))

    graph = tf.get_default_graph()
    coloc_DTD_T = tf.get_collection('DTD_T')

    activations = tf.get_collection('DTD')

    x_placeholder_new = activations[0]

    prediction = coloc_DTD_T[0]

    logits = coloc_DTD_T[1]
    y_placeholder_new = coloc_DTD_T[2]

    # y_placeholder_new = graph.get_tensor_by_name("Classifier/y_placeholder:0")

    # logits = graph.get_tensor_by_name("Classifier/CNN/layer5/l5logits/MatMul:0")

    # correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(y_placeholder_new, 1), name="equal_train_op1")
    train_accuracy = graph.get_tensor_by_name("Classifier/reduce_mean_op:0")

    print('testing account:', dataset._num_data)
    test_accuracy_scores = []
    # predict_scores = numpy.zeros(shape=(dataset._num_data, num_classes))
    # labelss = numpy.zeros(shape=(dataset._num_data, num_classes))

    predict_scores = []
    labelss = []
    i = 0
    sample_size = 0
    for data, labels in dataset.iter_batch(batch_size, 1):
        predict_score, test_accuracy_score = session.run([prediction, train_accuracy],
                                                         feed_dict={x_placeholder_new: data, y_placeholder_new: labels})
        print('test Accuracy:', test_accuracy_score)
        test_accuracy_scores.append(test_accuracy_score)

        # predict_score = session.run(prediction, feed_dict={x_placeholder_new: data, y_placeholder_new: labels})
        predict_scores.append(predict_score)
        labelss.append(labels)

        sample_size = sample_size + len(data)
        i += 1

    print("test sample number:%d" % sample_size)
    labelss = numpy.reshape(labelss, (sample_size, num_classes))
    predict_scores = numpy.reshape(predict_scores, (sample_size, num_classes))

    # predict_scores = numpy.array(predict_scores)
    # labelss = numpy.array(labelss)

    targetNames = ['class 0', 'class 1']
    y_test_argmax = labelss.argmax(axis=1)
    predict_argmax = predict_scores.argmax(axis=1)

    # print(classification_report(y_true=y_test_argmax, y_pred=predict_argmax, target_names=targetNames))

    num_list = [n for n in range(0, num_classes)]
    num_list_new = [str(x) for x in num_list]

    cnf_matrix = confusion_matrix(y_true=y_test_argmax, y_pred=predict_argmax)
    print("confusion matrix:/n")
    print(cnf_matrix)

    test_accuracy_scores = numpy.array(test_accuracy_scores)

    print("test_accuracy_score shape: ", test_accuracy_scores.shape)
    print("avag test_accuracy_score: %s" % test_accuracy_scores.mean())

    # Compute ROC curve and area the curve

    roc_auc = dict()

    cvscores = dict()
    tpr = dict()
    fpr = dict()
    tprs = dict()
    fprs = dict()
    aucs = dict()

    for i in range(num_classes):
        fpr[i], tpr[i], _ = roc_curve(labelss[:, i], predict_scores[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    fpr["micro"], tpr["micro"], _ = roc_curve(labelss.ravel(), predict_scores.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    plot_multi_class(fpr, tpr, roc_auc)

total_end_time = datetime.now()
print("/n total duration : {}".format(total_end_time, total_start_time))
