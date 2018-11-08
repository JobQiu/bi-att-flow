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
    import argparse
    parser = argparse.ArgumentParser()
    target_dir = "/Users/xavier.qiu/Documents/GitHub/bi-att-flow/data/squad"
    glove_dir = "/Users/xavier.qiu/Documents/GitHub/bi-att-flow/data/glove"  # os.path.join(home, "data", "glove")

    parser.add_argument('-s', '--source_dir', default=target_dir)
    parser.add_argument('-t', '--target_dir', default=target_dir)
    # what does action mean in parser add argument?
    # https://docs.python.org/3.8/library/argparse.html#action

    parser.add_argument("--glove_corpus", default="6B")
    parser.add_argument('-g', "--glove_dir", default=glove_dir)
    parser.add_argument("--glove_vec_size", default=100, type=int)
    parser.add_argument("--version", default="1.1", type=str)
    parser.add_argument("--train_ratio", default=.9, type=float)

    # not clear about these arguments

    parser.add_argument('-d', '--debug', action='store_true')
    parser.add_argument("--mode", default="full", type=str)
    parser.add_argument("--single_path", default="", type=str)
    parser.add_argument("--tokenizer", default="PTB", type=str)
    parser.add_argument("--url", default="vision-server2.corp.ai2", type=str)
    parser.add_argument("--port", default=8000, type=int)
    parser.add_argument("--split", action="store_true")
    args = parser.parse_args()
    return args


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
    elif args.mode == 'all':
        pass
    elif args.mode == "single":
        pass
    else:
        pass

    pass


def prepro_each(args, data_type, start_ratio=0.0, stop_ratio=1.0, out_name="default", in_path=None):
    """


    :param args:            arguments
    :param data_type:       train, dev or all
    :param start_ratio:     default is 0.0
    :param stop_ratio:      default is 1.0
    :param out_name:        train, dev or test
    :param in_path:         default is None, not sure about what is this
    :return:
    """

    # 1. tokenize and sent tokenize

    if args.tokenizer == "PTB":
        import nltk
        sent_tokenize = nltk.sent_tokenize

        def word_tokenize(tokens):
            """
            firstly word_tokenize the tokens and replace some
            chars, and return a list
            :param tokens:
            :return:
            """
            return [token.replace("''", '"').replace("``", '"') for token in nltk.word_tokenize(tokens)]
    elif args.tokenizer == 'Stanford':
        from my.corenlp_interface import CoreNLPInterface
        interface = CoreNLPInterface(args.url, args.port)
        sent_tokenize = interface.split_doc
        word_tokenize = interface.split_sent
    else:
        raise Exception()

    if not args.split:
        sent_tokenize = lambda para: [para]  # input is para, turn it to a list

    # 2. load data from disk
    source_path = in_path or os.path.join(args.source_dir,
                                          "{}-v{}.json".format(data_type, args.version))
    source_data = json.load(open(file=source_path, mode='r'))

    # 3. initiate some counter and some lists
    q, cq, rx, rcx = [], [], [], []
    # question, char_question, context, char_context
    y, cy, ids, idxs = [], [], [], []
    x, cx = [], []
    answerss, p = [], []
    word_counter, char_counter, lower_word_counter = Counter(), Counter(), Counter()
    start_at_index = int(round(len(source_data['data']) * start_ratio))
    stop_at_index = int(round(len(source_data['data']) * stop_ratio))

    # 4. iterate the dataset

    for article_index, article in enumerate(tqdm(source_data['data'][start_at_index:stop_at_index])):
        xp, cxp, pp = [], [], []
        x.append(xp)
        cx.append(cxp)
        p.append(pp)

        for paragraph_index, paragraph in enumerate(article['paragraphs']):
            context = paragraph['context']
            context = context.replace("''", '" ')
            # notice this space, so the length of the context will not change when replace
            context = context.replace("``", '" ')

            # context is a str here
            list_of_wordlist = list(map(word_tokenize, sent_tokenize(context)))
            # after sent_tokenizer, it will be a list of sentence, here just one sentence,
            # a list of sentence
            # then the map, will apply the word_tokenize func to each sentence
            # a list of lists of words
            # [[words for sentence1], [words for sentence2]]

            list_of_wordlist = [process_tokens(tokens) for tokens in list_of_wordlist]
            # list_of_wordlist is a 2d stuff

            list_of_charlist = [[list(word) for word in wordlist] for wordlist in list_of_wordlist]
            # list of charlist is a 3d, sentence-dim, word-dim, char-dim

            xp.append(list_of_wordlist)
            # 3d, paragraph, sentence, words
            cxp.append(list_of_charlist)
            # 4d, paragraph, sentence, words, chars
            pp.append(context)
            # 2d, paragraph, context

            ## update counters
            num_qas = len(paragraph['qas'])
            for wordlist in list_of_wordlist:
                for word in wordlist:
                    word_counter[word] += num_qas
                    lower_word_counter[word.lower()] += num_qas
                    for char in word:
                        char_counter[char] += num_qas

            rxi = [article_index, paragraph_index]
            assert len(x) - 1 == article_index
            # x stores xp, xp is 3d, paragraph, sentece, and words
            assert len(x[article_index]) - 1 == paragraph_index

            for question in paragraph['qas']:
                question_wordslist = word_tokenize(question['question'])
                # it's a list of words
                question_charslist = [list(word) for word in question_wordslist]
                # it's a list of charlist
                yi = []
                cyi = []
                answers = []  # the content of each answers

                for answer in question['answers']:
                    answer_text = answer['text']
                    answers.append(answer_text)
                    answer_start_index = answer['answer_start']
                    answer_end_index = answer_start_index + len(answer_text)
                    yi0, yi1 = get_word_span(context,
                                             list_of_wordlist,  # 2-d: sentences, words
                                             answer_start_index,
                                             answer_end_index)
                    # yi0 (0, 108), 0 is the index of sentence
                    # yi1 (0, 111). 108 and 111 is the start and end of word index

                    assert len(list_of_wordlist[yi0[0]]) > yi0[1]
                    # the length of the first sentence is larger than 108
                    assert len(list_of_wordlist[yi1[0]]) >= yi1[1]
                    # the length of the first sentence is larger or equla to 111

                    w0 = list_of_wordlist[yi0[0]][yi0[1]]  # the start words of the answer
                    w1 = list_of_wordlist[yi1[0]][yi1[1] - 1]  # the last word of the answer

                    i0 = get_word_idx(context, list_of_wordlist, yi0)
                    i1 = get_word_idx(context, list_of_wordlist, (yi1[0], yi1[1] - 1))
                    # i0 is 515, which is the char index of the answer,
                    # i1 is start index of the final word in terms of chars
                    # 'Saint Bernadette Soubirous', i1 is the index of S in Soubirous
                    cyi0 = answer_start_index - i0
                    # it should be 0 here since start index is 515, and i0 should also be 515
                    cyi1 = answer_end_index - i1 - 1
                    # cyi1 seems to be the length of last word -1, or because some other issues

                    assert answer_text[0] == w0[cyi0], (answer_text, w0, cyi0)
                    assert answer_text[-1] == w1[cyi1]
                    # be sure the first char and last char are same with the first word's first char and last word's last char
                    assert cyi0 < 32, (answer_text, w0)
                    assert cyi1 < 32, (answer_text, w1)
                    #
                    yi.append([yi0, yi1])  # index of words
                    cyi.append([cyi0, cyi1])
                    # index of shifts from the first char and last char of the answer in context

                # update counters
                for word in question_wordslist:
                    word_counter[word] += 1
                    lower_word_counter[word.lower()] += 1
                    for char in word:
                        char_counter[char] += 1

                q.append(question_wordslist)  # 2-d list of wordlist for each question
                cq.append(question_charslist)  # 3-d, question-word-char
                y.append(yi)  # question-startendpair
                cy.append(cyi)  # question-startend char pair
                rx.append(rxi)  # list of article_id-paragraph_id pair
                rcx.append(rxi)
                ids.append(question['id'])  # ids for each question
                idxs.append(len(idxs))  # index for each question
                answerss.append(answers)  # list of answer in string

    word2vec_dict = get_word2vec(args, word_counter)
    lower_word2vec_dict = get_word2vec(args, lower_word_counter)

    # add context here
    data = {
        'q': q,  # list of word list of each questions, [['who','are', 'you'], ... ]
        'cq': cq,
        # [<class 'list'>: [['T', 'o'], ['w', 'h', 'o', 'm'], ['d', 'i', 'd'], ['t', 'h', 'e'], ['V', 'i', 'r', 'g', 'i', 'n'], ['M', 'a', 'r', 'y'], ['a', 'l', 'l', 'e', 'g', 'e', 'd', 'l', 'y'], ['a', 'p', 'p', 'e', 'a', 'r'], ['i', 'n'], ['1', '8', '5', '8'], ['i', 'n'], ['L', 'o', 'u', 'r', 'd', 'e', 's'], ['F', 'r', 'a', 'n', 'c', 'e'], ['?']] , ...]
        'y': y,  # list of <class 'list'>: [[(0, 108), (0, 111)]]
        '*x': rx,  # list of <class 'list'>: [0, 21], 0 means the number of article, 21 means the 21st paragraph
        '*cx': rcx,  # same with rx but for characters, i guess the values are same as well
        'cy': cy,  #
        'idxs': idxs,  # just those ids
        'ids': ids,  # the id of each question, sth like uuid
        'answerss': answerss,  # the content of the answer
        '*p': rx  #
    }
    shared = {
        'x': x,  # words of each paragraph
        'cx': cx,  # characters of each
        'p': p,  # the content of each paragraph
        'word_counter': word_counter,
        'char_counter': char_counter,
        'lower_word_counter': lower_word_counter,
        'word2vec': word2vec_dict,
        'lower_word2vec': lower_word2vec_dict
    }

    print("saving ...")
    save(args, data, shared, out_name)


def get_word2vec(args, word_counter):
    """
    build a dictionary, key is the word, the value is the embedding of those words
    only those words in the word_counter will be appended into this dict's keys
    :param args:
    :param word_counter:
    :return:
    """

    glove_path = os.path.join(args.glove_dir,
                              "glove.{}.{}d.txt".format(args.glove_corpus, args.glove_vec_size))
    sizes = {
        '6B': int(4e5),
        '42B': int(1.9e6),
        '840B': int(2.2e6),
        '2B': int(1.2e6)
    }

    total = sizes[args.glove_corpus]
    word2vec_dict = {}
    with open(glove_path, 'r', encoding='utf-8') as fh:
        for line in tqdm(fh, total=total):
            # what is total in tqdm?
            array = line.lstrip().rstrip().split(" ")
            word = array[0]
            vector = list(map(float, array[1:]))

            if word in word_counter:
                word2vec_dict[word] = vector
            elif word.lower() in word_counter:
                word2vec_dict[word.lower()] = vector
            elif word.capitalize() in word_counter:
                word2vec_dict[word.capitalize()] = vector
            elif word.upper() in word_counter:
                word2vec_dict[word.upper()] = vector
    print("{}/{} of word vocab have corresponding vectors in {}".format(len(word2vec_dict),
                                                                        len(word_counter),
                                                                        glove_path))
    return word2vec_dict


def save(args, data, shared, data_type):
    """

    :param args:
    :param data:
    :param shared:
    :param data_type: train or dev
    :return:
    """

    data_path = os.path.join(args.target_dir, "data_{}.json".format(data_type))
    shared_path = os.path.join(args.target_dir, "shared_{}.json".format(data_type))
    json.dump(data, open(data_path, 'w'))
    json.dump(shared, open(file=shared_path, mode='w'))


if __name__ == "__main__":
    main()
