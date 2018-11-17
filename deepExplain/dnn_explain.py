from __future__ import print_function
import sys
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
            new_saver = tf.train.import_meta_graph(ckptdir + '.meta')
            new_saver.restore(sess, tf.train.latest_checkpoint(logdir))

            weights = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='CNN')
            opera = tf.get_collection('DTD_T')
            prediction = opera[0]
            logits = opera[1]
            y_placeholder_new = opera[2]

            activations = tf.get_collection('DTD')
            x_placeholder_new = activations[0]

            cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y_placeholder_new)
                                  )

            correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(y_placeholder_new, 1))
            train_accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

            with DeepExplain(session=sess) as de:
                attributions = {
                    # Gradient-based
                    # NOTE: reduce_max is used to select the output unit for the class predicted by the classifier
                    # For an example of how to use the ground-truth labels instead, see mnist_cnn_keras notebook
                    'Saliency maps': de.explain('saliency', logits, x_placeholder_new, sample_imgs),
                    'Gradient * Input': de.explain('grad*input', logits, x_placeholder_new, sample_imgs),
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
    #print("current script:", script_name)
    script_name = script_name[0:-3]  # remove ".py"

    return script_name

hmaps = []


fast_file = "../data/Q97V95_SULSO.txt"
# fast_file = "/home/myc/projectpy/cnnTensorflowNew/data/Q8EHI4_SHEON_5-69.txt"
# fast_file = "../data/Q6M020_METMP_269-320"
name = "Q97V95_SULSO"
aligned_seqs = get_seqs("/data1/projectpy/cnnTensorflow/data/PF00571_seed.txt", name)

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
ckptdir = "../taylorDecomposition/log/log_1542223750_fold_0/_model"
logdir = "../taylorDecomposition/log/log_1542223750_fold_0"
data, labels, seq = dataset.next_sample_random(with_raw=True)

rawseq = seq[0][1]
with open("../result/"+cur_script_name() + "_" + name + "_seq.txt", 'w') as f:
    # rawseq = seq[0][1].replace("_", "")
    f.write("%s\n" % rawseq)
    for item in list(rawseq):
        f.write("%s\n" % item)

hmaps = run_deep_explain(logdir, ckptdir, data, labels)

for key in hmaps:
    nhmaps3 = numpy.reshape(hmaps[key][0], (1000, 21)).mean(axis=1)
    for seqobj in aligned_seqs:
        start = seqobj.start
        end = seqobj.end
        seq = seqobj.seq
        location = seqobj.location

        relevance_score = numpy.zeros(len(seq))

        index = 0
        raw_index = 0
        for i, c in enumerate(seq):
            if c != '.':
                index = raw_index + start - 1
                raw_c = rawseq[index]

                if raw_c != c:
                    raise Exception("index is wrong")

                score = nhmaps3[index] * 1000

                relevance_score[i] = score

                raw_index = raw_index + 1

        with open("../result/"+cur_script_name() + "_" + name + "_" + key + "_" + location + "_seq.txt", 'w') as f:
            #        f.write("%s\n\n\n" % seqobj.location)
            for item in list(seq):
                f.write("%s\n" % item)
        with open("../result/"+cur_script_name() + "_" + name + "_" + key + "_" + location + "_score.txt", 'w') as f:
            #        f.write("%s\n\n\n" % seqobj.location)
            for item in list(relevance_score):
                f.write("%s\n" % item)
    print("relevance scores %s: " % key)
    print(nhmaps3.argsort()[-5:][::-1])
    print(nhmaps3[nhmaps3.argsort()[-5:][::-1]])


total_end_time = datetime.now()
print("/n/n total duration : {}".format(total_end_time, total_start_time))
