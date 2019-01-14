from __future__ import print_function
import sys
import os
import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

from datetime import datetime

from proteindataset import ProteinDataSet
from alignedSeqUtil import get_seqs

from deepexplain.tensorflow import DeepExplain

total_start_time = datetime.now()

batch_size = 100
num_classes = 2
num_features = 21000
collection_name = "sensitivity_analysis"
split_size = 3
epochs = 10


def model_summary():
    model_vars = tf.trainable_variables()
    slim.model_analyzer.analyze_vars(model_vars, print_info=True)


img_x, img_y = 1, num_features


def run_deep_explain(logdir, ckptdir, sample_imgs, label_imgs):
    hmaps = []
    with tf.Session() as sess:
        with DeepExplain(session=sess, graph=sess.graph) as de:
            ckpt = tf.train.latest_checkpoint(ckptdir)

            new_saver = tf.train.import_meta_graph(ckpt + '.meta')
            new_saver.restore(sess, tf.train.latest_checkpoint(logdir))

            # weights = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='CNN')
            opera = tf.get_collection('DTD_T')
            logits = opera[1]
            opera = tf.get_collection('DTD')
            x_placeholder_new = opera[0]
            # y_placeholder_new = opera[2]
            #
            # activations = tf.get_collection('DTD')
            # x_placeholder_new = activations[0]
            #
            model_summary()

            # cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y_placeholder_new)
            #                       )
            #
            # correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(y_placeholder_new, 1))
            # train_accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

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
                print("Done!")

    return attributions


def cur_script_name():
    argv0_list = sys.argv[0].split("/")
    script_name = argv0_list[len(argv0_list) - 1]  # get script file name self
    # print("current script:", script_name)
    script_name = script_name[0:-3]  # remove ".py"

    return script_name


def formatticks(x, pos):
    if x == '.':
        return pos
    else:
        return x


hmaps = []

fast_file = "../data/CaricaPapaya/MER0000647.txt"
# fast_file = "/home/myc/projectpy/cnnTensorflowNew/data/Q8EHI4_SHEON_5-69.txt"
# fast_file = "../data/Q6M020_METMP_269-320"
name = "MER0000647"
aligned_seqs = get_seqs("../data/CaricaPapaya/MER0000647_aligned.txt", name)

dataset = ProteinDataSet(fpath=fast_file,
                         seqlen=1000,
                         n_classes=2,
                         need_shuffle=False)
# ckptdir = "/home/myc/projectpy/cnnTensorflowNew/taylorDecomposition/log/log_1540842612_fold_0/_model"
# logdir = "/home/myc/projectpy/cnnTensorflowNew/taylorDecomposition/log/log_1540842612_fold_0"

# ckptdir = "/home/myc/projectpy/cnnTensorflowNew/taylorDecomposition/log/log_1540876682_fold_0/_model"
# logdir = "/home/myc/projectpy/cnnTensorflowNew/taylorDecomposition/log/log_1540876682_fold_0"

# ckptdir = "/home/myc/projectpy/cnnTensorflowNew/taylorDecomposition/log/log_1541007951_fold_0/_model"
# logdir = "/home/myc/projectpy/cnnTensorflowNew/taylorDecomposition/log/log_1541007951_fold_0"
# ckptdir = "./log/log_1541687626_fold_0/_model"
# logdir = "./log/log_1541687626_fold_0"
# ckptdir = "../taylorDecomposition/log/log_1542223750_fold_0/_model"
# logdir = "../taylorDecomposition/log/log_1542223750_fold_0"
# ckptdir = "/home/myc/projectpy/DeepFam/checkpoint/module.ckpt"
# logdir = "/home/myc/projectpy/DeepFam/checkpoint/"
ckptdir = "./log//log_1544202458_fold_4/"
logdir = "./log/log_1544202458_fold_4/"

data, labels, seq = dataset.next_sample_random(with_raw=True)

rawseq = seq[0][1]
with open(os.getcwd() + "/result/" + cur_script_name() + "_" + name + "_seq.txt", 'w') as f:
    # rawseq = seq[0][1].replace("_", "")
    f.write("%s\n" % rawseq)
    for item in list(rawseq):
        f.write("%s\n" % item)
muber_one = numpy.sum(data)

hmaps = run_deep_explain(logdir, ckptdir, data, labels)


def plot_score(seq, scores):
    scores = scores[0:len(seq)]
    plt.figure(figsize=(30, numpy.amax((scores)) + 0.05))
    plt.xlim(0)
    plt.ylim(0, numpy.amax((scores)))
    # fig1.set(figsize=(20, 2))
    plt.axhline(0, color='black')
    plt.tick_params(axis='x', rotation=90, labelsize=5)
    # fig1.set_xticklabels(fontsize=8)

    plt.bar(list(range(len(seq))), (scores),  tick_label=list(range(1, len(seq) + 1)), align='center')
    for a, b in zip(list(range(len(seq))), (scores)):
        if b > 0:
            plt.text(a, b + 0.05, seq[a], ha='center', va='bottom', fontsize=6)

    plt.tight_layout()
    plt.savefig(os.getcwd() + "/result/" + cur_script_name() + "_" + name + "_" + key + "_" + "raw.png")
    plt.close()


for key in hmaps:
    raw_indexs=[]
    nhmaps3 = numpy.reshape(hmaps[key][0], (1000, 21)).max(axis=1) * 1000
    plot_score(rawseq, nhmaps3)

    fig = plt.figure(figsize=(300, 2.5 * len(aligned_seqs)*3))

    for seq_index, seqobj in enumerate(aligned_seqs):
        start = seqobj.start
        end = seqobj.end
        seq = seqobj.seq
        location = seqobj.location

        relevance_score = numpy.zeros(len(seq))

        index = 0
        raw_index = 0
        for i, c in enumerate(seq):
            if c != '-':
                index = raw_index + start - 1
                raw_c = rawseq[index]

                if raw_c != c:
                    print("%s location is wrong,sequence is %s,but aligned is %s " % (index, raw_c, c))
                    raise Exception("index is wrong")

                score = nhmaps3[index]

                relevance_score[i] = score

                raw_index = raw_index + 1
                raw_indexs.append(index+1)
            else:
                raw_indexs.append('.')

        with open(os.getcwd() + "/result/" + cur_script_name() + "_" + name + "_" + key + "_" + location + "_seq.txt",
                  'w') as f:
            #        f.write("%s\n\n\n" % seqobj.location)
            for item in list(seq):
                f.write("%s\n" % item)
        with open(os.getcwd() + "/result/" + cur_script_name() + "_" + name + "_" + key + "_" + location + "_score.txt",
                  'w') as f:
            #        f.write("%s\n\n\n" % seqobj.location)
            for item in list(relevance_score):
                f.write("%s\n" % item)

        numpy.savetxt(
            os.getcwd() + "/result/" + cur_script_name() + "_" + name + "_" + key + "_" + location + "_hmaps.txt",
            nhmaps3, fmt="%5.5f")

        # formatter = matplotlib.ticker.FuncFormatter(formatticks)
        # locator = matplotlib.ticker.MaxNLocator(nbins=6)

        fig1 = fig.add_subplot(2, 1, seq_index + 1)
        fig1.set_xlim(0)
        fig1.set_ylim(0, numpy.amax(nhmaps3))
        # fig1.set(figsize=(20, 2))
        fig1.set_title(location)
        fig1.axhline(0, color='black')

        # fig1.xaxis.set_major_formatter(formatter)
        # fig1.xaxis.set_major_locator(locator)

        fig1.bar(list(range(len(seq))), relevance_score.tolist(), tick_label=seq, align='center')

        fig2 = fig.add_subplot(2, 1, 2)
        fig2.tick_params(axis='x', rotation=90)
        fig2.set_xlim(0)
        fig2.set_ylim(0, numpy.amax(nhmaps3))
        fig2.bar(list(range(len(seq))), relevance_score.tolist(), tick_label=raw_indexs, color='rgb')

    seq_list = list(rawseq)
    bigger_list = nhmaps3.argsort()[-30:][::-1]
    bigger_scores = nhmaps3[bigger_list]

    print("relevance scores %s: " % key)

    for i, item in enumerate(bigger_list):
        if (i + 1) % 10 == 0:
            print("%s:" % seq_list[item], end="")
            print("%s," % str(item + 1), end="")
            print("\n")
        else:
            print("%s:" % seq_list[item], end="")
            print("%s," % str(item + 1), end="")

    # print(nhmaps3.argsort()[-30:][::-1])
    # print(nhmaps3[nhmaps3.argsort()[-30:][::-1]])

    plt.tight_layout()
    plt.savefig(os.getcwd() + "/result/" + cur_script_name() + "_" + name + "_" + key + ".png")
    plt.close()

total_end_time = datetime.now()
print("\n total duration : {}".format(total_end_time, total_start_time))
