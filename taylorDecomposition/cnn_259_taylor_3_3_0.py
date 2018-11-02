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

from datetime import datetime

from batchDataset import Dataset

from models_2_4_0 import MNIST_CNN, Taylor

from dataset import DataSet


## output roc plot according to cross-validation,

total_start_time = datetime.now()

batch_size = 10
num_classes = 1074
num_features = 21000
collection_name = "sensitivity_analysis"
split_size = 3
epochs = 2

def get_Data():
    X = []
    Y = []
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

    print("feature shape:", X.shape)
    print("label shape:", Y.shape)

    return X, Y

def model_summary():
    model_vars = tf.trainable_variables()
    slim.model_analyzer.analyze_vars(model_vars, print_info=True)


img_x, img_y = 1, num_features

with tf.name_scope('Classifier'):
    # Initialize neural network
    DNN = MNIST_CNN('CNN')

    # Setup training process
    x_placeholder = tf.placeholder(tf.float32, [None, 21000], name='x_placeholder')
    y_placeholder = tf.placeholder(tf.float32, [None, 1074], name='y_placeholder')

    trainingFlag = tf.placeholder_with_default(True, shape=[], name='trainingFlag')

    activations, prediction, logits = DNN(x_placeholder)

    tf.add_to_collection('DTD', x_placeholder)

    for activation in activations:
        tf.add_to_collection('DTD', activation)

    tf.add_to_collection('DTD', prediction)

    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y_placeholder))

    optimizer = tf.train.AdamOptimizer().minimize(cost, var_list=DNN.vars)

    correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(y_placeholder, 1))
    train_accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    model_summary()


cost_summary = tf.summary.scalar('Train Cost', cost)
accuray_summary = tf.summary.scalar('Train Accuracy', train_accuracy)
summary = tf.summary.merge_all()

model_summary()

def run_taylor_decompostion(x_train, y_train, logdir, ckptdir):
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

    return hmaps


def run_sensitivity_analysis(checkpoint_infos, class_num, feature_num, collection_name):
    hmaps = numpy.zeros([class_num, feature_num], dtype=numpy.float32)

    for i, checkpoint_info in enumerate(checkpoint_infos):
        logdir, ckptdir, sample_imgs = checkpoint_info[0], checkpoint_info[1], checkpoint_info[2]

        tf.reset_default_graph()

        sess = tf.InteractiveSession()

        new_saver = tf.train.import_meta_graph(ckptdir + '.meta')
        new_saver.restore(sess, tf.train.latest_checkpoint(logdir))

        SA = tf.get_collection(collection_name)
        training = SA[0]
        new_x_placeholder = SA[1]
        new_y_pred = SA[2]

        # SA_scores = [tf.square(tf.gradients(new_y_pred[:, i], new_x_placeholder)) for i in range(class_num)]
        SA_scores = [new_x_placeholder * tf.gradients(new_y_pred[:, i], new_x_placeholder) for i in range(class_num)]

        hmap = numpy.reshape(
            [sess.run(SA_scores[i], feed_dict={new_x_placeholder: sample_imgs[i][None, :], training: False}) for i in
             range(class_num)], [class_num, feature_num])

        hmaps = hmaps + hmap

        sess.close()

    return hmaps / (len(checkpoint_infos))


def run_deep_taylor_decomposition(checkpoint_infos, class_num, feature_num, collection_name):
    hmaps = []

    for i, checkpoint_info in enumerate(checkpoint_infos):
        logdir, ckptdir, sample_imgs = checkpoint_info[0], checkpoint_info[1], checkpoint_info[2]

        tf.reset_default_graph()

        sess = tf.InteractiveSession()

        new_saver = tf.train.import_meta_graph(ckptdir + '.meta')
        new_saver.restore(sess, tf.train.latest_checkpoint(logdir))

        weights = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='CNN')
        activations = tf.get_collection('DTD')
        x_placeholder_new = activations[0]

        conv_ksize = [1, 3, 3, 1]
        pool_ksize = [1, 2, 2, 1]
        conv_strides = [1, 1, 1, 1]
        pool_strides = [1, 2, 2, 1]

        weights.reverse()
        activations.reverse()

        taylor = Taylor(activations, weights, conv_ksize, pool_ksize, conv_strides, pool_strides, 'Taylor')

        Rs = []
        for i in range(2):
            Rs.append(taylor(i))

        imgs = []
        for i in range(2):
            imgs.append(sess.run(Rs[i], feed_dict={x_placeholder_new: sample_imgs[i][None, :]}))
        hmaps.append(imgs)

        sess.close()
    return hmaps


def run_test(session, x_test, y_test, cvscores, tprs, fprs, aucs, fold):
    print('testing account:', len(x_test))
    test_accuracy_score = session.run(train_accuracy, feed_dict={x_placeholder: x_test, y_placeholder: y_test})
    print('test Accuracy:', test_accuracy_score)

    predict_score = session.run(prediction, feed_dict={x_placeholder: x_test, y_placeholder: y_test})

    targetNames = ['class 0', 'class 1']
    y_test_argmax = argmax(y_test, axis=1)
    predict_argmax = argmax(predict_score, axis=1)

    print(classification_report(y_true=y_test_argmax, y_pred=predict_argmax, target_names=targetNames))

    cnf_matrix = confusion_matrix(y_test_argmax, predict_argmax)

    print(cnf_matrix)

    print('roc:%.2f%%' % roc_auc_score(y_test_argmax, predict_argmax))

    # Compute ROC curve and area the curve
    fpr, tpr, thresholds = roc_curve(y_test[:, 1], predict_score[:, 1])

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
    script_num = script_name.split('_')[2]

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


def run_train(session, dataset, logdir, ckptdir, epochs):
    print("\n Start training")

    saver = tf.train.Saver()
    file_writer = tf.summary.FileWriter(logdir, tf.get_default_graph())

    for epoch in range(epochs):
        total_batch = int(dataset._num_data / batch_size)
        avg_cost = 0
        avg_acc = 0

        for data, labels in dataset.iter_once(batch_size):

            _, c, a, summary_str = session.run([optimizer, cost, train_accuracy, summary],
                                               feed_dict={x_placeholder: data, y_placeholder: labels})
            avg_cost += c / total_batch
            avg_acc += a / total_batch

        file_writer.add_summary(summary_str, epoch * total_batch )

        print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.9f}'.format(avg_cost), 'accuracy =',
              '{:.9f}'.format(avg_acc))

        saver.save(session, ckptdir)
    # print('Accuracy:', session.run(accuracy, feed_dict={X: mnist.test.images, Y: mnist.test.labels}))


def plot_precision_recall_vs_threshold(precisions, recalls, thresholds, fold):
    plt.plot(thresholds, precisions[:-1], label='Precision fold %d' % fold)
    plt.plot(thresholds, recalls[:-1], label='Recall fold %d' % fold)


def  cross_validate(session, dataset, epochs, class_num, feature_num, collection_name, split_size=5):
    results = []
    hmaps = []
    cvscores = dict()
    tprs = dict()
    fprs = dict()
    aucs = dict()
    checkpoints = []

    seed = 123456

    current_time = time.time()
    time_tag = str(int(current_time))


    checkpoint = []

    logdir = sys.path[0] + "/log/log_" + time_tag + "_fold_"
    ckptdir = logdir + '_model'

    if not os.path.exists(ckptdir):
        os.makedirs(ckptdir)

    checkpoint.append(logdir)
    checkpoint.append(ckptdir)

    start_time = datetime.now()

    print("training data count:", dataset._num_data)

    # train
    run_train(session, dataset, logdir, ckptdir, epochs)
    # print('Accuracy:', session.run(train_accuracy, feed_dict={x_placeholder: x_test, y_placeholder: y_test}))
    # test
    # test_accuracy_score, predict_score = run_test(session, x_test, y_test, cvscores, tprs, fprs, aucs, fold)
    # results.append(test_accuracy_score)
    #
    # precisions, recalls, thresholds = precision_recall_curve(argmax(y_test, axis=1), argmax(predict_score, axis=1))
    # plot_precision_recall_vs_threshold(precisions, recalls, thresholds, fold)
    #
    # # taylor decompostion
    # images = x_train
    # labels = y_train
    #
    # sample_img = []
    # for i in range(num_classes):
    #     sample_img.append(images[numpy.argmax(labels, axis=1) == i][3])
    #
    # checkpoint.append(sample_img)
    #
    # checkpoints.append(checkpoint)
    #
    #     # hmap = run_taylor_decompostion(x_train, y_train, logdir, ckptdir)
    #     # hmaps.append(hmap)
    #
    # end_time = datetime.now()
    #
    # print("The ", fold, " fold Duration: {}".format(end_time - start_time))
    # # draw precision recall plot and then close plt
    # plt.xlabel("Threshold")
    # plt.legend(loc='lower right')
    # plt.ylim([0, 1])
    # plt.savefig('./' + cur_script_name() + '_precision_recall')
    # plt.close()

    # draw plot
    plot_draw_cross_validation(cvscores, tprs, fprs, aucs)

    return results, checkpoints


hmaps = []

with tf.Session() as session:
    #X, Y = get_Data()

    train_file = "/data1/projectpy/DeepFam/data/COG-500-1074/90percent/data_all.txt"

    dataset = DataSet(fpath=train_file,
                      seqlen=1000,
                      n_classes=1074,
                      need_shuffle=True)

    #feature, label = dataset.full_batch()

    session.run(tf.global_variables_initializer())

    results, checkpoints = cross_validate(session, dataset, epochs, num_classes, num_features,
                                          collection_name, split_size)

    # print("Cross-validation test accuracy: %s" % results)
    # print("Test accuracy: %f" % session.run(accuracy, feed_dict={x_placeholder: test_x, y_placeholder: test_y}))

hmaps = run_deep_taylor_decomposition(checkpoints, num_classes, num_features, collection_name)

nhmaps = numpy.array(hmaps)
numpy.save(cur_script_name()+"hmaps", nhmaps)

print("relevance scores: %s" % hmaps)

total_end_time = datetime.now()
print("/n/n total duration : {}".format(total_end_time, total_start_time))
