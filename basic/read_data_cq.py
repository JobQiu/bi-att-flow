import itertools
import json
import math
import os
import random
from collections import defaultdict
import numpy as np

from my.tensorflow import grouper
from my.utils import index


class Data(object):
    pass


class DataSet(object):

    def __init__(self, data, data_type, shared=None, valid_idxs=None):
        """

        :param data:
        :param data_type:
        :param shared:
        :param valid_idxs:
        """
        pass


def read_data(config, data_type, load, data_filter=None):
    """
    train_data = read_data(config, 'train', config.load, data_filter=data_filter)

    for example, we will use the above code to load the dataset,
    and return a DataSet object that wrapped it and that dataSet object will be used later

    :param config:
    :param data_type:
    :param load:
    :param data_filter:
    :return: a DataSet instance
    """

    # 1. load data and share
    data_path = os.path.join(config.data_dir, "data_{}.json".format(data_type))
    shared_path = os.path.join(config.data_dir, "shared_{}.json".format(data_type))
    with open(data_path, 'r') as fh:
        data = json.load(fh)
    with open(shared_path, 'r') as fh:
        shared = json.load(fh)

    # to get the number of questions, we just need to pick one kind from the data, and count its length
    num_questions = len(next(iter(data.values())))

    # 2. filter questions
    if data_filter is None:
        valid_idxs = range(num_questions)
        pass
    else:
        mask = []
        keys = data.keys()
        values = data.values()
        for vals in zip(*values):
            # the consequence of this for, is
            # each value for each type,
            each = {key: val for key, val in zip(keys, vals)}
            mask.append(data_filter(each, shared))
            pass
        valid_idxs = [idx for idx in range(len(mask)) if mask[idx]]
        # if mask is true, we will put the index in to valid idxs
    print("Loaded {}/{} examples {}".format(len(valid_idxs), num_questions, data_type))

    shared_path = config.shared_path or os.path.join(config.out_dir, 'shared.json')
    # when we saved it?

    # 3. load share.json or build one
    if not os.path.isfile(shared_path) or not load:
        if config.lower_word:
            word2vec_dict = shared['lower_word2vec']
            word_counter = shared['lower_word_counter']
        else:
            word2vec_dict = shared['word2vec']
            word_counter = shared['word_counter']
        char_counter = shared['char_counter']

        if config.finetune:
            pass
        else:
            assert config.known_if_glove
            assert config.use_glove_for_unk

            shared['word2idx'] = {word: idx + 2 for idx, word in enumerate(
                word for word, count in word_counter.items() if
                count > config.word_count_th and word not in word2vec_dict)}

        shared['char2idx'] = {char: idx + 2 for idx, char in
                              enumerate(char for char, count in char_counter if count > config.char_count_th)}

        # here put all the words and chars whose count is larger than the threshold into this word2idx and char2idx
        NULL = "-NULL-"
        UNK = "-UNK-"
        shared['word2idx'][NULL] = 0
        shared['word2idx'][UNK] = 1
        shared['char2idx'][NULL] = 0
        shared['char2idx'][UNK] = 1
        json.dump({'word2idx': shared['word2idx'], 'char2idx': shared['char2idx']}, open(shared_path, 'w'))

        pass
    else:
        # if there is no shared file we left before, or we don't want to load it,
        # we are gonna to create it and write stuff in it
        new_shared = json.load(open(shared_path, 'r'))
        for key, val in new_shared.items():
            shared[key] = val

        pass

    # 4.
    # if config.use_glove_for_unk:
    #    word2vec_dict = shared['lower_word2vec'] if
    pass
