import tensorflow as tf
import numpy as np
import tensorflow.contrib.slim as slim

import re


def inference(data, seq_len, num_features, num_classes, window_lengths, num_windows, num_hidden, keep_prob, regularizer,
              for_training=False, scope=None):
    pred, layers = network(data, seq_len, num_features, num_classes, window_lengths, num_windows, num_hidden, keep_prob,
                           regularizer,
                           is_training=for_training,
                           scope=scope)

    return pred, layers


def get_placeholders(num_feature, num_classes):
    placeholders = {}
    placeholders['data'] = tf.placeholder(tf.float32, [None, num_feature])
    placeholders['labels'] = tf.placeholder(tf.float32, [None, num_classes])
    return placeholders


def network(data, seq_len, num_features, num_classes, window_lengths, num_windows, num_hidden, keep_prob, regularizer,
            is_training=True, scope=''):
    batch_norm_params = {
        'decay': 0.9,  # might problem if too small updates
        'is_training': is_training,
        'updates_collections': None
    }

    layers = {}

    x_data = tf.reshape(data, [-1, 1, seq_len * num_features, 1])

    ###
    # define layers
    ###

    with tf.name_scope(scope, 'v1', [x_data]):
        with slim.arg_scope([slim.conv2d, slim.fully_connected],
                            activation_fn=tf.nn.relu,
                            biases_initializer=tf.constant_initializer(0.1),
                            normalizer_fn=slim.batch_norm,
                            normalizer_params=batch_norm_params,
                            weights_initializer=tf.contrib.layers.xavier_initializer()):
            layers['conv'] = []
            layers['conv2'] = []
            layers['hidden1'] = []
            layers['hidden3'] = []
            for i, wlen in enumerate(window_lengths):
                layers['conv'].append(slim.conv2d(x_data,
                                                  num_windows[i],
                                                  [1, wlen],
                                                  # padding='SAME',
                                                  padding='VALID',
                                                  weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                                  scope="conv%d" % i))
                # max pooling
                max_pooled = slim.max_pool2d(layers['conv'][i],
                                             # [1, FLAGS.seq_len],
                                             [1, 3],
                                             stride=[1, 1],
                                             padding='VALID',
                                             scope="pool%d" % i)
                # reshape
                layers['hidden1'].append(slim.flatten(max_pooled, scope="flatten%d" % i))

            # concat
            layers['concat'] = tf.concat(layers['hidden1'], 1)

            # dropout
            dropped = slim.dropout(layers['concat'],
                                   keep_prob=keep_prob,
                                   is_training=is_training,
                                   scope="dropout")

            # fc layers
            layers['hidden2'] = slim.fully_connected(dropped,
                                                     num_hidden,
                                                     weights_regularizer=None,
                                                     scope="fc1")

            layers['pred'] = slim.fully_connected(layers['hidden2'],
                                                  num_classes,
                                                  activation_fn=None,
                                                  normalizer_fn=None,
                                                  normalizer_params=None,
                                                  weights_regularizer=slim.l2_regularizer(
                                                      regularizer) if regularizer > 0 else None,
                                                  scope="fc2")

    return layers['pred'], layers
