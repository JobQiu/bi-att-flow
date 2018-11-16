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
    """

    :param config:
    :return:
    """
    models = []
    for gpu_index in range(config.num_gpus):
        # tf name scope, is a context manager for use when defining a python op
        # tf device return a context manager that specifies the default device to use for newly created ops
        # so these ops are under the certain name_scope and device.
        with tf.name_scope("model_{}".format(gpu_index)) as scope, tf.device(
                "/{}:{}".format(config.device_type, gpu_index)):
            model = Model(config, scope, rep=gpu_index == 0)
            tf.get_variable_scope().reuse_variables()

            # todo what is get variable scope.
            # return the current variable scope, is this the name scope, or what.
            # todo what is tf scope reuse variables
            # if you do want the variables to be shared, you have two options,
            # you can set reuse when create the variable or set scope.reuse_variable
            models.append(model)
    return models


class Model(object):

    def __init__(self, config, scope, rep=True):
        """

        :param config:
        :param scope:
        :param rep:
        """

        self.scope = scope
        self.config = config
        # tf get variable, gets a new variable if exist
        self.global_step = tf.get_variable('global_step',
                                           shape=[],
                                           dtype='int32',
                                           initializer=tf.constant_initializer,
                                           trainable=False)

        batch_size = config.batch_size
        W = config.max_word_size  # the max length of each word, here it's 16

        self.x = tf.placeholder('int32', [batch_size, None, None], name='x')
        self.cx = tf.placeholder('int32', [batch_size, None, None, W], name='cx')
        self.x_mask = tf.placeholder('bool', [batch_size, None, None], name='x_mask')

        self.q = tf.placeholder('int32', [batch_size, None], name='q')
        self.cq = tf.placeholder('int32', [batch_size, None, W], name='cq')
        self.q_mask = tf.placeholder('bool', [batch_size, None], name='q_mask')

        # todo 1. what are these masks, x_mask and q_mask

        self.y = tf.placeholder('bool', [batch_size, None, None], name='y')
        self.y2 = tf.placeholder('bool', [batch_size, None, None], name='y2')
        # todo 2. why y and y2 are boolean, what do they mean

        self.is_train = tf.placeholder('bool', [], name='is_train')

        self.new_emb_mat = tf.placeholder('float', [None, config.word_emb_size], name="new_emb_mat")
        # todo 3. what is the embedding matrix mean
        # i guess this one is used to reduce the size of the vocabulary

        self.tensor_dict = {}  # define misc
        # todo 4. what is misc?

        self.logits = None
        self.y_predict = None  # y_predict
        self.var_list = None

        self.loss = None
        self._build_forward()
        self._build_loss()
        self.var_ema = None
        # todo 4. what is ema?

        # todo what is rep
        if rep:
            self._build_var_ema()

        if config.mode == 'train':
            self._build_ema()

        self.summary = tf.summary.merge_all()
        # todo what is tf.get collection?
        self.summary = tf.summary.merge(tf.get_collection("summaries", scope=self.scope))

    def _build_forward(self):
        """

        :return:
        """

        config = self.config
        maxSentPerPara = config.max_num_sents
        maxCharPerWord = config.max_word_size
        maxWordPerSent = config.max_sent_size
        maxWordPerQues = config.max_ques_size
        char_emb_size = config.char_emb_size
        char_kinds = config.char_vocab_size
        word_kinds = config.word_vocab_size
        hidden_size = config.hidden_size
        char_word_emb_size = config.char_out_size
        word_emb_size = config.word_emb_size
        # the first scope word and char embedding
        with tf.variable_scope("emb"):

            if config.use_char_emb:
                # create a matrix to store the embedding to map each char to a 8-long vector, for this example, the shape is 330 * 8
                with tf.variable_scope('emb_var'), tf.device('/cpu:0'):
                    char_emb_mat = tf.get_variable("char_emb_mat",
                                                   shape=[char_kinds, char_emb_size],
                                                   dtype='float')

                with tf.variable_scope('char'):
                    Acx = tf.nn.embedding_lookup(char_emb_mat, self.cx)
                    '''
                    the shape of self.cx is [batch_size, number of sentence, number of word each sentence, 
                    number of char each word, the vector for each char], 5-dim
                    '''

                    Acq = tf.nn.embedding_lookup(char_emb_mat, self.cq)
                    '''
                    the shape is 4-dim, because only one sentence for each question '''
                    Acx = tf.reshape(Acx, [-1, maxWordPerSent, maxCharPerWord, char_emb_size])
                    # here it seems like the author assume there are only one sentence for each paragraph
                    Acq = tf.reshape(Acq, [-1, maxWordPerQues, maxCharPerWord, char_emb_size])

                    filter_sizes = list(map(int, config.out_channel_dims.split(',')))
                    heights = list(map(int, config.filter_heights.split(',')))

                    assert sum(filter_sizes) == char_word_emb_size, (filter_sizes, char_word_emb_size)
                    # this means the sum of these filters will output a vector such as 100

                    # do the conv to construct the char word embedding
                    with tf.variable_scope('conv'):
                        xx = multi_conv1d(Acx,
                                          filter_sizes,
                                          heights,
                                          "VALID",
                                          self.is_train,
                                          config.keep_prob,
                                          scope="xx")
                        if config.share_cnn_weights:
                            tf.get_variable_scope().reuse_variables()
                            qq = multi_conv1d(Acq,
                                              filter_sizes,
                                              heights,
                                              "VALID",
                                              self.is_train,
                                              config.keep_prob,
                                              scope='qq')
                        else:
                            qq = multi_conv1d(Acq,
                                              filter_sizes,
                                              heights,
                                              "VALID",
                                              self.is_train,
                                              config.keep_prob,
                                              scope='qq')
                        xx = tf.reshape(xx, [-1, maxSentPerPara, maxWordPerSent, char_word_emb_size])
                        qq = tf.reshape(qq, [-1, maxWordPerQues, char_word_emb_size])

            # end of char_word embedding

            if config.use_word_emb:
                with tf.variable_scope('emb_var'), tf.device("/cpu:0"):
                    if config.mode == 'train':
                        word_emb_mat = tf.get_variable("word_emb_mat", dtype='float',
                                                       shape=[word_kinds, word_emb_size],
                                                       initializer=get_initializer(config.emb_mat))
                        # not clear about why get initializer work here
                    else:
                        word_emb_mat = tf.get_variable("word_emb_mat", dtype='float',
                                                       shape=[word_kinds, word_emb_size])
                    if config.use_glove_for_unk:
                        word_emb_mat = tf.concat([word_emb_mat, self.new_emb_mat], 0, )

                    # todo not clear about this part, why we need to use self new emb mat?

                with tf.name_scope("word"):
                    Ax = tf.nn.embedding_lookup(word_emb_mat, self.x)
                    Aq = tf.nn.embedding_lookup(word_emb_mat, self.q)

                    self.tensor_dict['x'] = Ax
                    self.tensor_dict['q'] = Aq

                if config.use_char_emb:
                    xx = tf.concat([xx, Ax], 3)
                    qq = tf.concat([qq, Aq], 2)
                else:
                    xx = Ax
                    qq = Aq

        # end of word embedding

        if config.highway:
            with tf.variable_scope("highway"):
                # todo, read and implement this highway network
                xx = highway_network(arg=xx,
                                     num_layers=config.highway_num_layers,
                                     bias=True,
                                     wd=config.wd,  # wd means weight decay
                                     is_train=self.is_train)
                tf.get_variable_scope().reuse_variables()
                qq = highway_network(arg=qq,
                                     num_layers=config.higyway_num_layer,
                                     bias=True,
                                     wd=config.wd,
                                     is_train=self.is_train)

        self.tensor_dict['xx'] = xx
        self.tensor_dict['qq'] = qq

        cell = BasicLSTMCell(num_units=hidden_size,
                             state_is_tuple=True)
        dropout_cell = SwitchableDropoutWrapper(cell,
                                                self.is_train,
                                                input_keep_prob=config.input_keep_prob)
        x_len = tf.reduce_mean(tf.cast(self.x_mask, 'int32'), 2)
        q_len = tf.reduce_mean(tf.cast(self.q_mask, 'int32'), 1)
        # todo what is x mask and q mask?

        with tf.variable_scope('prepro'):
            (forward_unit, backward_unit), ((_, forward_unit_final), (_, backward_unit_final)) = \
                bidirectional_dynamic_rnn(dropout_cell,
                                          qq, q_len,
                                          dtype='float',
                                          scope='u1')
            # todo what is u1, the first layer,
            u = tf.concat([forward_unit, backward_unit], 2)

            if config.share_lstm_weights:
                tf.get_variable_scope().reuse_variables()
                (forward_hidden_state, backward_hidden_state), _ = bidirectional_dynamic_rnn(cell, cell, xx, x_len,
                                                                                             dtype='float', scope='u1')
            else:
                (forward_hidden_state, backward_hidden_state), _ = bidirectional_dynamic_rnn(cell, cell, xx, x_len,
                                                                                             dtype="float", scope='h1')

            # todo why xx is cell and qq is drop cell?
            h = tf.concat([forward_hidden_state, backward_hidden_state], 3)
            self.tensor_dict['u'] = u
            self.tensor_dict['h'] = h

        # end of convert qq and xx to lstm output hidden states

        with tf.variable_scope('main'):
            if config.dynamic_att:
                # todo what is dynamic attention?
                p0 = h  # h is context, u is questions
                u = tf.reshape(tf.tile(tf.expand_dims(u, 1)))


                pass
            else:
                pass

            pass

        pass

    def _build_loss(self):
        """

        :return:
        """
        pass

    def _build_ema(self):
        """
        
        :return: 
        """
        pass

    def _build_var_ema(self):
        """

        :return:
        """
        pass

    def get_loss(self):
        return self.loss

    def get_global_step(self):
        return self.global_step

    def get_var_list(self):
        return self.var_list

    def get_feed_dict(self, batch, is_train, supervised=True):
        """
        
        :param batch: 
        :param is_train: 
        :param supervised: 
        :return: 
        """
        pass


def bi_attention(config, is_train, h, u, h_mask=None, u_mask=None, scope=None, tensor_dict=None):
    """

    :param config:
    :param is_train:
    :param h:
    :param u:
    :param h_mask:
    :param u_mask:
    :param scope:
    :param tensor_dict:
    :return:
    """

    pass


def attention_layer(config, is_train, h, u, h_mask=None, u_mask=None, scope=None, tensor_dict=None):
    """

    :param config:
    :param is_train:
    :param h:
    :param u:
    :param h_mask:
    :param u_mask:
    :param scope:
    :param tensor_dict:
    :return:
    """
    pass
