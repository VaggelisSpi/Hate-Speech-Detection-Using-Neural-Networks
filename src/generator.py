import numpy as np
from nltk import word_tokenize


class KerasBatchGenerator(object):
    def __init__(self, data, classes, dataset_size, batch_size, num_classes, vectorizer, vec_size, cycle=True):
        """
        Constructor of the class KerasBatchGenerator. This generator is used when we don't 
        have a word embeddings layer.

        :param data: The whole dataset
        :param classes: An array or list with the class of each element
        :param dataset_size: The size of the dataset
        :param batch_size: The size each batch will have
        :param num_classes: The number of the classes that we want to categorize our data
        :param vectorizer: A class that is used to vectorize the data. It should contain a function named vectorize that
                           takes a single data element and returns an n-dimensional array.
        :param: cycle: Indicate if we will cycle to the beginning to fill a batch if no more data left or we will reset
                        the data to the begging
        :param vec_size: The dimensionality of the each data point
        """
        self.data = data
        self.classes = classes
        self.dataset_size = dataset_size
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.vectorizer = vectorizer
        self.vec_size = vec_size
        self.cycle = cycle
        # Value to pad elements with less tokens than the max length of each batch
        self.pad_value = -100.0
        # this will track the progress of the batches sequentially through the
        # data set - once the data reaches the end of the data set, it will
        # reset back to zero
        self.current_idx = 0

    def generate(self):
        """
        Returns a batch of the dataset with size = self.batch_size. The data are vectorized.

        :return: Returns 2 np-arrays x and y. X is the array with the vectorized data and its shape is
                (batch_size, maxlen, vec_size) where maxlen is the number of tokens of the largest element in each
                batch. It is possible that the last batch will have less elements if the batch size does not divide
                exactly the size of the dataset. Y is the array that holds the classes of each element of the batch and
                its shape is (batch_size, num_classes)
        """
        while True:
            # If we reached the end of the dataset then, if cycle is set to true fill the batch with the remaining
            # data and go back in the beginning to fill the rest. Otherwise start from the beginning
            if self.current_idx + self.batch_size > self.dataset_size:
                if self.cycle:
                    # Get the length of the longest element in the dataset
                    maxlen = len(word_tokenize(self.data[self.dataset_size - 1]))

                    cur_batch_size = self.batch_size

                    # Initialize the arrays we will return
                    x = np.zeros((cur_batch_size, maxlen, self.vec_size))
                    y = np.zeros((cur_batch_size, self.num_classes))

                    # Iterate through the remaining the data of the current batch and vectorize it
                    for j in range(self.current_idx, self.dataset_size):
                        # Vectorize the current data element
                        x[j - self.current_idx:] = self.vectorizer.vectorize(self.data[j], maxlen, self.pad_value)
                        y[j - self.current_idx] = self.classes[j]

                    end_j = self.dataset_size - self.current_idx
                    # Fill the rest of the batch with data from the beginning
                    for j in range(self.batch_size - end_j):
                        x[j + end_j:] = self.vectorizer.vectorize(self.data[j], maxlen, self.pad_value)
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

            # Get the length of the longest element in the current batch
            maxlen = len(word_tokenize(self.data[cur_batch_end - 1]))

            # Initialize the arrays we will return
            x = np.zeros((cur_batch_size, maxlen, self.vec_size))

            # Iterate through the data of the current batch and vectorize it
            for j in range(cur_batch_start, cur_batch_end):
                # Vectorize the current data element
                x[j - cur_batch_start:] = self.vectorizer.vectorize(self.data[j], maxlen, self.pad_value)

            # Get the classes for the current batch
            y = np.array(self.classes[cur_batch_start:cur_batch_end])

            # Update current index for the next iteration if there is data left
            self.current_idx += self.batch_size

            yield x, y

    def __next__(self):
        """
        Returns a batch of the dataset with size = self.batch_size. The data are vectorized.

        :return: Returns 2 np-arrays x and y. X is the array with the vectorized data and its shape is
                (batch_size, maxlen, vec_size) where maxlen is the number of tokens of the largest element in each
                batch. It is possible that the last batch will have less elements if the batch size does not divide
                exactly the size of the dataset. Y is the array that hold the classes of each element of the batch and
                its shape is (batch_size, num_classes)
        """
        return self.generate
