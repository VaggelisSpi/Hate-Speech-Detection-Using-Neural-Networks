import numpy as np
import pandas as pd
import math
import utils
from scipy.sparse import csr_matrix
from tensorflow.keras.utils import Sequence
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical


class wordEmbeddingsGenerator:
    def __init__(self, data, classes, dataset_size, batch_size, num_classes, cycle, vocab_size, word_to_id, maxlen):
        """
        Constructor of the class wordEmbeddingsGenerator. This generator is used when we 
        have a word embeddings layer.

        :param data: The whole dataset. An array or list
        :param classes: An array or list with the class of each element
        :param dataset_size: The size of the dataset
        :param batch_size: The size each batch will have
        :param num_classes: The number of the classes that we want to categorize our data
        :param: cycle: Indicate if we will cycle to the beginning to fill a batch if no more data left or we will reset
                        the data to the begging
        :param vocab_size: The number of unique words
        :param word_to_id: A dictionary that maps words to ids
        """
        self.data = data
        self.classes = classes
        self.dataset_size = dataset_size
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.cycle = cycle
        self.vocab_size = vocab_size
        self.word_to_id = word_to_id
        self.maxlen = maxlen
        # this will track the progress of the batches sequentially through the
        # data set - once the data reaches the end of the data set it will reset
        # back to zero
        self.current_idx = 0

    def words_to_ids(self, text):
        """
        Gets a text as an input and returns a list with all the words mapped to their ids
        """
        ids = []
        tokens = text.split()
        for token in tokens:
            ids.append(self.word_to_id[token])
        
        length = len(ids)
        ids_array = np.zeros(self.maxlen)
        for i in range (length):
            ids_array[i] = ids[i]
        return ids_array

    def generate(self):
        """
        Returns a batch of the dataset with batch size = self.batch_size. The data are the ids of the words.

        :return: Returns 2 np-arrays x and y. X is the array/list with the data and its shape is (batch_size, z) where z
                 is the length of each element. It is possible that the last batch will have less elements if the batch size does
                 not divide exactly the size of the dataset. Y is the array that hold the classes of each element of the
                 batch and its shape is (batch_size, num_classes). If the cycle parameter is set to true then if the last batch
                 will have less elements that the batch size then it will be filled with data from the beginning of the dataset and
                 the rest of the batches will continue from where we were left
        """
        while True:
            # If we reached the end of the dataset then, if cycle is set to true fill the batch with the remaining
            # data and go back in the beginning to fill the rest. Otherwise start from the beginning
            i = 0
            if self.current_idx + self.batch_size > self.dataset_size:
                if self.cycle:
                    cur_batch_size = self.batch_size

                    # Initialize the arrays we will return
                    x = np.zeros((cur_batch_size, self.maxlen))
                    y = np.zeros((cur_batch_size, self.num_classes))

                    # Iterate through the remaining data of the current batch and vectorize it
                    for j in range(self.current_idx, self.dataset_size):
                        # Vectorize the current data element
                        x[j - self.current_idx] = self.words_to_ids(self.data[j])
                        y[j - self.current_idx] = self.classes[j]

                    end_j = self.dataset_size - self.current_idx
                    # Fill the rest of the batch with data from the beginning
                    for j in range(self.batch_size - end_j):
                        x[j + end_j] = self.words_to_ids(self.data[j])
                        y[j + end_j] = self.classes[j]

                    # Update current index for the next iteration if there is data left
                    self.current_idx = self.batch_size - end_j

                    yield x, y
                    continue
                else:
                    # The batch will start from the beginning
                    cur_batch_size = self.batch_size
                    cur_batch_start = 0
                    self.current_idx = 0
            else:
                cur_batch_size = self.batch_size
                cur_batch_start = self.current_idx

            cur_batch_end = cur_batch_start + cur_batch_size  # Ending index of the current batch

            # Initialize the arrays we will return
            x = np.zeros((cur_batch_size, self.maxlen))

            # Iterate through the data of the current batch and vectorize it
            for j in range(cur_batch_start, cur_batch_end):
                # Vectorize the current data element
                x[j - cur_batch_start] = self.words_to_ids(self.data[j])

            # Get the classes for the current batch
            y = np.array(self.classes[cur_batch_start:cur_batch_end])

            # Update current index for the next iteration if there is data left
            self.current_idx += self.batch_size

            yield x, y

    def __next__(self):
        """
        Returns a batch of the dataset with size = self.batch_size. The data are vectorized.

        :return: Returns 2 np-arrays x and y. X is the array with the vectorized data and its shape is
                (batch_size, self.vec_size, vec_size) where self.vec_size is the number of tokens of the largest
                element in each batch. It is possible that the last batch will have less elements if the batch size does
                not divide exactly the size of the dataset. Y is the array that hold the classes of each element of the
                batch and its shape is (batch_size, num_classes)
        """
        return self.generate()
