from __future__ import print_function
import sys
import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

from datetime import datetime

from models_2_4_1 import MNIST_CNN, Taylor
from proteindataset import ProteinDataSet

## output roc plot according to cross-validation,

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


def run_deep_taylor_decomposition(logdir, ckptdir, sample_imgs, label_imgs):
    hmaps = []
    with tf.Session() as sess:
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

        # Rs = []
        # for i in range(num_classes):
        #     Rs.append(taylor(i))
        label_taylor = taylor(numpy.argmax(label_imgs))

        hmaps.append(sess.run(label_taylor, feed_dict={x_placeholder_new: sample_imgs}))

    return hmaps


def cur_script_name():
    argv0_list = sys.argv[0].split("/")
    script_name = argv0_list[len(argv0_list) - 1]  # get script file name self
    print("current script:", script_name)
    script_name = script_name[0:-3]  # remove ".py"

    return script_name


def read_fasta(fasta_file, maxlen, familynumber):
    input = open(fasta_file, 'r')

    chrom_seq = ''
    chrom_id = None

    label = numpy.zeros((1, num_classes), dtype=numpy.uint8)
    label[int(familynumber)] = 1

    for line in input:
        if line[0] == '>':
            if chrom_id is not None:
                yield chrom_id, chrom_seq

            chrom_seq = ''
            chrom_id = line.split()[0][1:].replace("|", "_")
        else:
            chrom_seq += line.strip().upper()

    input.close()

    padded_seq = seq + "_" * (maxlen - len(seq))

    return label, padded_seq


hmaps = []

train_file = "/home/myc/projectpy/DeepFam/data/COG-500-1074/90percent//data_388.txt"
fast_file = "../data/Q8EHI4_SHEON.txt"
#fast_file = "/home/myc/projectpy/cnnTensorflowNew/data/Q8EHI4_SHEON_5-69.txt"
#fast_file = "../data/Q6M020_METMP_269-320"
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
ckptdir = "./log/log_1541687626_fold_0/_model"
logdir = "./log/log_1541687626_fold_0"

data, labels, seq = dataset.next_sample_random(with_raw=True)

hmaps = run_deep_taylor_decomposition(logdir, ckptdir, data, labels)

nhmaps = numpy.array(hmaps)
numpy.save(cur_script_name() + "hmaps", nhmaps)

nhmaps2 = numpy.reshape(nhmaps, (1000, 21))
nhmaps3 = nhmaps2.sum(axis=1)
numpy.savetxt(cur_script_name() + "hmaps.txt", nhmaps3 * 1000, fmt="%5.5f")
# f = open(cur_script_name() + "seq.txt", 'w')
# f.writelines(["%s\n" % item for item in list(seq)])

with open(cur_script_name() + "_seq.txt", 'w') as f:
    #rawseq = seq[0][1].replace("_", "")
    rawseq = seq[0][1]
    f.write("%s\n" % rawseq)
    for item in list(rawseq):
        f.write("%s\n" % item)

print(seq)
# print(labels)
print(nhmaps3)

print(nhmaps3.argsort()[-5:][::-1])
print(nhmaps3[nhmaps3.argsort()[-5:][::-1]])
print("relevance scores: ", nhmaps)
print("relevance scores: ", hmaps)

total_end_time = datetime.now()
print("/n/n total duration : {}".format(total_end_time, total_start_time))
