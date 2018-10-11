from __future__ import print_function
import sys
import os
import time

import keras
import tensorflow as tf

from sklearn.metrics import classification_report, roc_auc_score, roc_curve, make_scorer, confusion_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_curve, auc

import numpy
from numpy import argmax
from scipy import interp
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

from datetime import datetime

from batchDataset import Dataset


def tp(y_true, y_pred): return confusion_matrix(y_true, y_pred)[0, 0]


def tn(y_true, y_pred): return confusion_matrix(y_true, y_pred)[0, 0]


def fp(y_true, y_pred): return confusion_matrix(y_true, y_pred)[1, 0]


def fn(y_true, y_pred): return confusion_matrix(y_true, y_pred)[0, 1]


total_start_time = datetime.now()


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


batch_size = 50
num_classes = 2
num_features = 259
collection_name = "sensitivity_analysis"

img_x, img_y = 1, num_features

x_placeholder = tf.placeholder(tf.float32, [None, img_y])
y_placeholder = tf.placeholder(tf.float32, [None, num_classes], name="truth")

trainingFlag = tf.placeholder_with_default(True, shape=[], name='training')

reshaped_input = tf.reshape(x_placeholder, [-1, img_x, img_y, 1], name="absolute_input")

y_pred = tf.keras.layers.Conv2D(filters=128, kernel_size=[3, 3], strides=(1, 1), padding='same',
                                activation='relu')(reshaped_input)
y_pred = tf.keras.layers.MaxPool2D(pool_size=(1, 1), strides=(1, 1))(y_pred)

y_pred = tf.keras.layers.Flatten()(y_pred)
y_pred = tf.keras.layers.Dense(256, activation='relu')(y_pred)

y_pred = tf.keras.layers.Dense(num_classes, activation=tf.nn.sigmoid, name="absolute_output")(y_pred)

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=y_pred, labels=y_placeholder))
# loss = tf.reduce_mean(
#     tf.keras.backend.categorical_crossentropy(target=y_placeholder, output=y_placeholder, from_logits=True
#                                               ))

train_optim = tf.train.AdamOptimizer().minimize(loss)

correct_prediction = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y_placeholder, 1))
train_accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

tf.add_to_collection(collection_name, trainingFlag)
tf.add_to_collection(collection_name, x_placeholder)
tf.add_to_collection(collection_name, y_pred)

cost_summary = tf.summary.scalar('Train Cost', loss)
accuray_summary = tf.summary.scalar('Train Accuracy', train_accuracy)
summary = tf.summary.merge_all()


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

        #SA_scores = [tf.square(tf.gradients(new_y_pred[:, i], new_x_placeholder)) for i in range(class_num)]
        SA_scores = [new_x_placeholder * tf.gradients(new_y_pred[:, i], new_x_placeholder) for i in range(class_num)]

        hmap = numpy.reshape(
            [sess.run(SA_scores[i], feed_dict={new_x_placeholder: sample_imgs[i][None, :], training: False}) for i in
             range(class_num)], [class_num, feature_num])

        hmaps = hmaps + hmap

        sess.close()

    return hmaps / (len(checkpoint_infos))


def plot_draw(cvscores, tprs, aucs):
    mean_fpr = numpy.linspace(0, 1, 100)

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


def run_train(session, x_train, y_train, logdir, ckptdir, epochs):
    print("\n Start training")
    init = tf.global_variables_initializer()
    session.run(init)

    train_data = Dataset(x_train, y_train)

    saver = tf.train.Saver()
    file_writer = tf.summary.FileWriter(logdir, tf.get_default_graph())

    for epoch in range(epochs):
        total_batch = int(train_data._num_examples / batch_size)
        avg_cost = 0
        avg_acc = 0

        for i in range(total_batch):
            batch_x, batch_y = train_data.next_batch(batch_size)
            _, c, a, summary_str = session.run([train_optim, loss, train_accuracy, summary],
                                               feed_dict={x_placeholder: batch_x, y_placeholder: batch_y})
            avg_cost += c / total_batch
            avg_acc += a / total_batch

            file_writer.add_summary(summary_str, epoch * total_batch + i)

        print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.9f}'.format(avg_cost), 'accuracy =',
              '{:.9f}'.format(avg_acc))

        saver.save(session, ckptdir)


def run_test(session, x_test, y_test, cvscores, tprs, aucs):
    test_accuracy_score = session.run(train_accuracy, feed_dict={x_placeholder: x_test, y_placeholder: y_test})
    print('test Accuracy:', test_accuracy_score)

    predict_score = session.run(y_pred, feed_dict={x_placeholder: x_test, y_placeholder: y_test})

    scoring = {'tp': make_scorer(tp), 'tn': make_scorer(tn), 'fp': make_scorer(fp), 'fn': make_scorer(fn)}

    targetNames = ['class 0', 'class 1']
    y_test_argmax = argmax(y_test, axis=1)
    predict_argmax = argmax(predict_score, axis=1)

    print(classification_report(y_test_argmax, predict_argmax, target_names=targetNames))

    cnf_matrix = confusion_matrix(y_test_argmax, predict_argmax)

    print(cnf_matrix)

    print('roc:%.2f%%' % roc_auc_score(y_test_argmax, predict_argmax))

    # Compute ROC curve and area the curve
    fpr, tpr, thresholds = roc_curve(y_test[:, 1], predict_score[:, 1])

    mean_fpr = numpy.linspace(0, 1, 100)

    tprs.append(interp(mean_fpr, fpr, tpr))
    tprs[-1][0] = 0.0
    roc_auc = auc(fpr, tpr)
    aucs.append(roc_auc)

    scores = predict_score
    cvscores.append(scores[1] * 100)

    return test_accuracy_score, tprs, aucs, cvscores


def cross_validate(session, X, Y, epochs, class_num, feature_num, collection_name, split_size=5):
    results = []
    hmaps = []
    cvscores = []
    tprs = []
    aucs = []
    checkpoints = []

    seed = 123456

    kfold = StratifiedKFold(n_splits=split_size, shuffle=True, random_state=seed)
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
        ky_train = Y[train]

        x_test = X[test]

        print("training data count:", len(x_train))

        x_train = x_train.astype('float32')
        x_test = x_test.astype('float32')

        y_train = keras.utils.to_categorical(ky_train, num_classes)
        y_test = keras.utils.to_categorical(Y[test], num_classes)
        # train
        run_train(session, x_train, y_train, logdir, ckptdir, epochs)
        # test
        test_accuracy = run_test(session, x_test, y_test, cvscores, tprs, aucs)
        results.append(test_accuracy)

        # draw plot
        plot_draw(cvscores, tprs, aucs)

        # taylor decompostion
        images = x_train
        labels = y_train

        sample_img = []
        for i in range(num_classes):
            sample_img.append(images[numpy.argmax(labels, axis=1) == i][3])

        checkpoint.append(sample_img)

        checkpoints.append(checkpoint)

        # hmap = run_taylor_decompostion(x_train, y_train, logdir, ckptdir)
        # hmaps.append(hmap)

        end_time = datetime.now()

        print("The ", fold, " fold Duration: {}".format(end_time - start_time))

    return results, checkpoints


hmaps = []
with tf.Session() as session:
    X, Y = get_Data()
    split_size = 3
    epochs = 10
    results, checkpoints = cross_validate(session, X, Y, epochs, num_classes, num_features,
                                          collection_name, split_size)

    # print("Cross-validation test accuracy: %s" % results)
    # print("Test accuracy: %f" % session.run(accuracy, feed_dict={x_placeholder: test_x, y_placeholder: test_y}))

hmap = run_sensitivity_analysis(checkpoints, num_classes, num_features, collection_name)
hmaps.append(hmap)
print("relevance scores: %s" % hmaps)

total_end_time = datetime.now()
print("/n/n total duration : {}".format(total_end_time, total_start_time))
