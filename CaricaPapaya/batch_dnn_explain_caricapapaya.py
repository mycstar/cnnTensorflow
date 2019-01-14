from __future__ import print_function
import sys
import os
import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

from datetime import datetime

from deepexplain.tensorflow import DeepExplain
from featureData import get_data
from models_2_4_caricapapaya import MNIST_CNN, Taylor

total_start_time = datetime.now()

num_features = 21000
seq_length = 1000
num_label = 2

## ######################## ##
#
#  Define CHARSET, CHARLEN
#
## ######################## ##
CHARSET = {'A': 0, 'C': 1, 'D': 2, 'E': 3, 'F': 4, 'G': 5, 'H': 6, \
           'I': 7, 'K': 8, 'L': 9, 'M': 10, 'N': 11, 'P': 12, 'Q': 13, \
           'R': 14, 'S': 15, 'T': 16, 'V': 17, 'W': 18, 'Y': 19, 'X': 20, \
           'O': 20, 'U': 20,
           'B': (2, 11),
           'Z': (3, 13),
           'J': (7, 9)}
CHARLEN = 21


## ######################## ##
#
#  Encoding Helpers
#
## ######################## ##
def encoding_seq_np(seq, arr):
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


def encoding_label_np(l, arr):
    arr[int(l)] = 1


def model_summary():
    model_vars = tf.trainable_variables()
    slim.model_analyzer.analyze_vars(model_vars, print_info=True)


def parse_data(seq, label, with_raw=False):
    encoded_data = np.zeros(CHARLEN * seq_length, dtype=np.float32)
    encoded_label = np.zeros(num_label, dtype=np.uint8)
    raw = seq

    encoding_label_np(label, encoded_label)
    encoding_seq_np(seq, encoded_data)

    if with_raw:
        return encoded_data, encoded_label, raw
    else:
        return encoded_data, encoded_label


def parse_data_batch(datas, with_raw=False):
    isize = len(datas)

    encoded_data = np.zeros((isize, CHARLEN * seq_length), dtype=np.float32)
    encoded_label = np.zeros((isize, num_label), dtype=np.uint8)
    raw = []

    for i, idx in enumerate(datas):
        label, seq = idx

        encoding_label_np(label, encoded_label[i])
        encoding_seq_np(seq, encoded_data[i])
        raw.append((label, seq))

    if with_raw:
        return encoded_data, encoded_label, raw
    else:
        return encoded_data, encoded_label


img_x, img_y = 1, num_features


def run_deep_explain(logdir, ckptdir, sample_imgs, label_imgs):
    with tf.Session() as sess:
        with DeepExplain(session=sess, graph=sess.graph) as de:
            ckpt = tf.train.latest_checkpoint(ckptdir)

            new_saver = tf.train.import_meta_graph(ckpt + '.meta')
            new_saver.restore(sess, tf.train.latest_checkpoint(logdir))

            weights = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='CNN')
            opera = tf.get_collection('DTD_T')
            logits = opera[1]
            activations = tf.get_collection('DTD')
            x_placeholder_new = activations[0]

            model_summary()

            with DeepExplain(session=sess) as de:
                attributions = {
                    # Gradient-based
                    # NOTE: reduce_max is used to select the output unit for the class predicted by the classifier
                    # For an example of how to use the ground-truth labels instead, see mnist_cnn_keras notebook
                    'Saliency maps': de.explain('saliency', logits, x_placeholder_new, sample_imgs),
                    'Gradient Input': de.explain('grad*input', logits, x_placeholder_new, sample_imgs),
                    'Integrated Gradients': de.explain('intgrad', logits, x_placeholder_new, sample_imgs),
                    'Epsilon-LRP': de.explain('elrp', logits, x_placeholder_new, sample_imgs),
                    'DeepLIFT (Rescale)': de.explain('deeplift', logits, x_placeholder_new, sample_imgs),
                    # Perturbation-based (comment out to evaluate, but this will take a while!)
                    # 'Occlusion [15x15]':    de.explain('occlusion', tf.reduce_max(logits, 1), X, xs, window_shape=(15,15,3), step=4)
                }
                print("DeepExplain Done!")

            conv_ksize = [1, 5, 5, 1]
            pool_ksize = [1, 2, 2, 1]
            conv_strides = [1, 1, 1, 1]
            pool_strides = [1, 2, 2, 1]

            weights.reverse()
            activations.reverse()

            taylor = Taylor(activations, weights, conv_ksize, pool_ksize, conv_strides, pool_strides, 'Taylor')
            label_taylor = taylor(np.argmax(label_imgs))

            res = sess.run(label_taylor, feed_dict={x_placeholder_new: sample_imgs})
            attributions['deep taylor'] = res

    return attributions


def run_deep_taylor_decomposition(logdir, ckptdir, sample_imgs, label_imgs):
    hmaps = []
    with tf.Session() as sess:
        ckpt = tf.train.latest_checkpoint(os.path.dirname(ckptdir))

        new_saver = tf.train.import_meta_graph(ckpt + '.meta')
        new_saver.restore(sess, tf.train.latest_checkpoint(logdir))

        weights = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='CNN')
        activations = tf.get_collection('DTD')
        x_placeholder_new = activations[0]

        conv_ksize = [1, 5, 5, 1]
        pool_ksize = [1, 2, 2, 1]
        conv_strides = [1, 1, 1, 1]
        pool_strides = [1, 2, 2, 1]

        weights.reverse()
        activations.reverse()

        taylor = Taylor(activations, weights, conv_ksize, pool_ksize, conv_strides, pool_strides, 'Taylor')

        # Rs = []
        # for i in range(num_classes):
        #     Rs.append(taylor(i))
        label_taylor = taylor(np.argmax(label_imgs))

        hmaps.append(sess.run(label_taylor, feed_dict={x_placeholder_new: sample_imgs}))

    return hmaps

def cur_script_name():
    argv0_list = sys.argv[0].split("/")
    script_name = argv0_list[len(argv0_list) - 1]  # get script file name self
    # print("current script:", script_name)
    script_name = script_name[0:-3]  # remove ".py"

    return script_name


def batch_sample(batch1):
    res = []
    for i, feature_data in enumerate(batch1):
        res.append((1, feature_data[1].getSeq()))
    return res


hmaps = []

ckptdir = "./log/log_1543854085_fold_4/"
logdir = "./log/log_1543854085_fold_4/"

feature_datas = get_data()

feature_list = []
for key in feature_datas.keys():
    feature_list.append((key, feature_datas.get(key)))

num_data = len(feature_list)
batch1 = feature_list[:80]

sample_datas1 = batch_sample(batch1)
datas, labels, raws = parse_data_batch(sample_datas1, with_raw=True)

hmaps_batch1 = run_deep_explain(logdir, ckptdir, datas, labels)

for key in hmaps_batch1:
    nhmaps3 = np.reshape(hmaps_batch1[key], (len(hmaps_batch1[key]), 1000, 21)).sum(axis=2) * 1000
    nhmaps_sums = nhmaps3.sum(axis=1)
    relevance_score_avg_list = []
    activate_avg_list = []

    for i, nhmaps_sum in enumerate(nhmaps_sums):
        sample_name, sample_feature = feature_list[i]
        sample_seq = sample_feature.getSeq()
        seq_length = len(sample_seq)
        relevance_score_avg = float(nhmaps_sum) / float(seq_length)
        relevance_score_avg_list.append(relevance_score_avg)

        sample_nhmaps = nhmaps3[i]
        valid_sample_nhmaps = [item for item in nhmaps3[i] if item > 0]
        activateSites = sample_feature.getActivateSites()
        activateSites = [item - 1 for item in activateSites]
        activate_scores = sample_nhmaps[activateSites]

        activate_avg = np.mean(activate_scores)
        activate_avg_list.append(activate_avg)

    plt.plot(list(range(len(relevance_score_avg_list))), relevance_score_avg_list, color='red', label='avg relevance')
    plt.plot(list(range(len(activate_avg_list))), activate_avg_list, color='blue', label='avg activate')
    plt.tight_layout()
    plt.savefig(os.getcwd() + "/result/" + cur_script_name() + "_" + key + ".png")
    plt.close()

total_end_time = datetime.now()
print("/n/n total duration : {}".format(total_end_time, total_start_time))
