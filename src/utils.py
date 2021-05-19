import numpy as np
from nltk import word_tokenize
import tensorflow as tf
import collections


def read_dictionary(dict_file):
    """
    Read a file of the form key value and make a dictionary out of it
    """
    _dict = {}
    f = open(dict_file, "r", encoding="utf-8")

    for _, line in enumerate(f):
        # TODO: Make this generic for multi-word entries
        tokens = line.split()
        word = tokens[0]
        values = np.asarray(tokens[1:], dtype='float32')

        _dict[word] = values

    return _dict


def dict_to_file(_dict, dict_file):
    """
    Create a file of the form key value from a given dictionary
    """
    f = open(dict_file, "w", encoding="utf-8")

    for key in _dict:
        f.write(key + " " + str(_dict[key]) + "\n")


def sample_floats(low=-0.5, high=13.3, size=(50,)):
    """
    Return a k-length list of unique random floats in the range of low <= x <= high.

    :param low:
    :param high:
    :param size:
    :return:
    """

    return np.random.uniform(low=low, high=high, size=size)


def tweet_len(row):
    """
    Count the tokens each tweet have. Tokenization is done using the nltk word_tokenize function

    :param row: A data frame row
    :return: An integer which is the number of tokens of the tweet
    """
    return len(word_tokenize(row["tweet"]))

def build_vocab(data):
    """
    Make a dictionary that maps words to id for the text provided
    """
    counter = collections.Counter(data)
    count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))

    words, _ = list(zip(*count_pairs))
    word_to_id = dict(zip(words, range(len(words))))

    return word_to_id

def make_whole_text(data):
    """
    Make a single string that contains all the data
    """
    wholeText = ''
    for text in data:
        wholeText = wholeText + ' ' + text

    return wholeText
