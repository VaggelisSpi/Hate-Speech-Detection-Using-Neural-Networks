import contextlib
import generator as gen
import my_models
import math
import numpy as np
import pandas as pd
import sys

from keras.callbacks import ModelCheckpoint
from sklearn.metrics import classification_report, confusion_matrix
from utils import tweet_len

def words_to_ids(text, word_to_id, maxlen):
        """
        Gets a text as an input and returns a list with all the words mapped to their ids
        """
        ids = []
        tokens = text.split()

        for token in tokens:
            ids.append(word_to_id[token])
        
        length = len(ids)
        ids_array = np.zeros(maxlen)

        for i in range (length):
            ids_array[i] = ids[i]
        
        for i in range(length, maxlen):
            ids_array[i] = 0

        return ids_array

@contextlib.contextmanager
def smart_open(filename=None):
    if filename and filename != '-':
        fh = open(filename, 'w')
    else:
        fh = sys.stdout

    try:
        yield fh
    finally:
        if fh is not sys.stdout:
            fh.close()


def print_report(result, y_true, y_pred, target_names, file_name='-'):
    """
    Write the classification results in a file
    """
    with smart_open(file_name) as f:
        print('Classification report:', file=f)
        print(classification_report(y_pred=y_pred, y_true=y_true, target_names=target_names), file=f)
        print('Confusion matrix:', file=f)
        print(confusion_matrix(y_true, y_pred), file=f)
        print('', file=f)
        print("Error and accuracy:", file=f)
        print(result, file=f)


def run_workflow(x_train, y_train, x_test, y_test, layers, 
                 go_backwards, use_embeddings, batch_size, vec_size, epochs,
                 vectorizer, model_name, train_generator, test_generator,
                 target_names, word_to_id, maxlen):
    """
    Make a model, train it, evaluate it and print the reports to a file

    :param x_train: The training data
    :param y_train: The classes of training data
    :param x_test: The testing data
    :param y_test: The classes of testing data
    :param layers: Number of LSTM layers of our model
    :param go_backwards: If true then build a bidirectional model
    :param use_embeddings: If set to true we will construct a model with an embeddings layer
    :param batch_size: Size of the training and testing batches to feed to the model
    :param epochs: Epochs to train the model for
    :param vectorizer: A class that takes raw data and produces vectorized data.
                       To be used by the generators
    :param model_name: A string that will be used as part of the file name for
                       the checkpoints and the report
    :param train_generator: A python generator that generates the training data set
    :param test_generator: A python generator that generates the testing data set
    :param target_names: A list of strings. To be used by the make_report function
    """
    model = my_models.make_model(layers=layers, go_backwards=go_backwards, use_embeddings=use_embeddings, batch_size=batch_size, vec_size=vec_size, input_length=maxlen)
    print(model.summary())

    train_size = len(x_train)
    test_size = len(x_test)

    print("Training the model")
    checkpointer = ModelCheckpoint(filepath='../MyModels/' + model_name + '-{epoch:02d}.hdf5', verbose=0)
    model.fit(train_generator.generate(), steps_per_epoch=math.ceil(train_size / batch_size),
              epochs=epochs, callbacks=[checkpointer])


    print("Evaluating the model")
    result = model.evaluate(test_generator.generate(), steps=math.ceil(test_size / batch_size), batch_size=batch_size)

    # Because our test data set does not divide exactly the batch size we will
    # extend it with data from the beggining so it can fit exactly the batch
    # size. We do this in order to make predictions for all the data. We will 
    # then remove the extra data and make our reports with the initial test data
    test_data_tweets = []

    if use_embeddings:
        for tweet in x_test:
            test_data_tweets.append(words_to_ids(tweet, word_to_id, maxlen))

        for i in range(test_size, math.ceil(test_size / batch_size)*batch_size):
            tweet = x_test.iloc[i - test_size]
            test_data_tweets.append(words_to_ids(tweet, word_to_id, maxlen))
    else:
        max_len = len(x_test.iloc[-1])
    
        for tweet in x_test:
            test_data_tweets.append(vectorizer.vectorize(tweet, maxlen=max_len))

        for i in range(test_size, math.ceil(test_size / batch_size)*batch_size):
            tweet = x_test.iloc[i - test_size]
            test_data_tweets.append(vectorizer.vectorize(tweet, maxlen=max_len))

    test_data_tweets = np.array(test_data_tweets)

    # Make the predictions
    print("Making the predictions")
    steps=math.ceil(test_size / batch_size)
    preds = model.predict(test_data_tweets, batch_size=batch_size, verbose=1, steps=steps)

    y_true = np.argmax(y_test[:test_size], axis=1)
    y_pred = np.argmax(preds[:test_size], axis=1)

    # Print the report
    print("Making the report")
    print_report(result, y_true, y_pred, target_names, '../MyReports/' + model_name + '_report.txt')