import numpy as np
from nltk import word_tokenize
# import gensim
# from gensim.models import Word2Vec
# import random
from operator import add
import utils
# import pandas as pd


class WordEmbeddingsVectorizer:
    def __init__(self, **kwargs):
        self.embeddings_file = kwargs.get("embeddings_file", None)
        self.vec_size = kwargs.get("vec_size", 200)
        self.embeddings_dict = {}

        if self.embeddings_file:
            self.make_embeddings()

    def make_embeddings(self):
        """
        Read the file with the embeddings

        :return:
        """
        self.embeddings_dict = utils.read_dictionary(self.embeddings_file)

    def vectorize(self, text, maxlen=0, pad_value=0.0):
        """
        Gets a single sentence and generate an array of vec_size.
        We generate an embedding for each token.
        In case a token does not exist in our dictionary we generate a random embedding for it.

        :param text: A single sentence to be vectorized
        :param maxlen: Maximum length of the current element to vectorize. If the current element is larger than maxlen
                       then we will truncate. If it's smaller then we fill the rest with the pad_value argument. If
                       maxlen is 0 then we leave the embedding as is
        :param pad_value: A value to fill the embedding in case it's smaller than maxlen
        :return: An n x maxlen-dimensional np array of floats. n is the number of tokens of the text and max_len is
                 the maximum size of each embedding. If max_len is 0 then the returned array would be of size n x
                 len(tokens)
        """
        vec = []

        # Tokenize the text
        tokens = word_tokenize(text)

        # If the text is empty make a random vector
        if not tokens:
            return utils.sample_floats(-0.5, 13.3, (self.vec_size,))

        # Iterate over all the tokens, generate an embedding for each one and add it to a list
        for token in tokens[:]:
            if token in self.embeddings_dict:
                vec.append(self.embeddings_dict[token])
            else:
                # Generate a random embedding for a token that does not exist in our dictionary
                vec.append(utils.sample_floats(-0.5, 13.3, (self.vec_size,)))

        # Pad the vector with noise
        if maxlen != 0:
            if len(vec) < maxlen:
                for i in range(len(vec), maxlen):
                    vec.append(np.full(self.vec_size, pad_value))
            elif len(vec) > maxlen:
                vec = vec[:maxlen]

        # Return the embeddings list as a numpy array
        return np.array(vec)