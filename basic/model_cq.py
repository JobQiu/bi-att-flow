import itertools
import random

import numpy as np
import tensorflow as tf
from tensorflow.python.ops.rnn_cell import BasicLSTMCell

from basic.read_data import DataSet
from my.tensorflow import get_initializer
from my.tensorflow.nn import softsel, get_logits, highway_network, multi_conv1d
from my.tensorflow.rnn import bidirectional_dynamic_rnn
from my.tensorflow.rnn_cell import SwitchableDropoutWrapper, AttentionCell


def get_multi_gpu_models(config):
    models = []
    for gpu_index in range(config.num_gpus):
        with tf.name_scope("model_{}".format(gpu_index)) as scope, tf.device(
                "/{}:{}".format(config.device_type, gpu_index)):
            #

            pass

    return models
