
import argparse
import json
import os
from collections import Counter
from tqdm import tqdm
from squad.utils import get_word_span, get_word_idx, process_tokens



def main():
    args = get_args()
    prepro(args)


def get_args():
    """
    the function used to set arguments
    :return:
    """
    parser = argparse.ArgumentParser()
    target_dir = "/Users/xavier.qiu/Documents/GitHub/bi-att-flow/data/squad"
    glove_dir = "/Users/xavier.qiu/Documents/GitHub/bi-att-flow/data/glove"  # os.path.join(home, "data", "glove")
    parser.add_argument('-s', "--source_dir", default=target_dir)
    parser.add_argument('-t', "--target_dir", default=target_dir)
    parser.add_argument('-d', "--debug", action='store_true')
    parser.add_argument("--train_ratio", default=0.9, type=int)
    parser.add_argument("--glove_corpus", default="6B")
    parser.add_argument("--glove_dir", default=glove_dir)
    parser.add_argument("--glove_vec_size", default=100, type=int)
    parser.add_argument("--mode", default="full", type=str)
    parser.add_argument("--single_path", default="", type=str)
    parser.add_argument("--tokenizer", default="PTB", type=str)
    parser.add_argument("--url", default="vision-server2.corp.ai2", type=str)
    parser.add_argument("--port", default=8000, type=int)
    parser.add_argument("--split", action='store_true')
    # TODO : put more args here
    return parser.parse_args()


# %%

def prepro(args):
    """
    preprocess the given dataset according to the configurations

    :param args:
    :return:
    """
    if not os.path.exists(args.target_dir):
        os.makedirs(args.target_dir)
    if args.mode == 'full':
        prepro_each(args, 'train', out_name='train')
        prepro_each(args, 'dev', out_name='dev')
        prepro_each(args, 'dev', out_name='test')

    pass


def prepro_each(args, data_type, start_ratio=0.0, stop_ratio=1.0, out_name="default", in_path=None):
    """


    :param args:
    :param data_type:
    :param start_ratio:
    :param stop_ratio:
    :param out_name:
    :param in_path:
    :return:
    """

    # 1. tokenize and sent tokenize

    if args.tokenizer == "PTB":
        import nltk
        sent_tokenize = nltk.sent_tokenize

        def word_tokenize(tokens):
            return [token.replace("''", '"').replace("``", '"') for token in nltk.word_tokenize(tokens)]
    elif args.tokenizer == 'Stanford':
        from my.corenlp_interface import CoreNLPInterface
        interface = CoreNLPInterface(args.url, args.port)
        sent_tokenize = interface.split_doc
        word_tokenize = interface.split_sent
    else:
        raise Exception()

    if not args.split:
        sent_tokenize = lambda para: [para]

    # 2. load data from disk
    source_path = in_path or os.path.join(args.source_dir, "{}-v2.0.json".format(data_type))
    source_data = json.load(open(source_path,'r'))

    # 3. initiate some counter and some lists
    q, cq, y, rx, rcx, ids, idxs = [], [], [], [], [], [], []
    cy = []
    x, cx = [], []
    answerss = []
    p = []
    word_counter, char_counter, lower_word_counter = Counter(), Counter(), Counter()
    start_at_index = int(round(len(source_data['data']) * start_ratio))
    stop_at_index = int(round(len(source_data['data']) * stop_ratio))

    # 4. iterate the dataset

    pass


if __name__ == "__main__":
    main()
