# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.11.2
#   kernelspec:
#     display_name: ThesisEnv
#     language: python
#     name: thesisenv
# ---

from google.colab import drive
drive.mount("/content/drive")

# cd "/content/drive/My Drive/Thesis_Project/src"

# !pip install emoji

# +
# Imports
import pandas as pd
import nltk
import tensorflow as tf
import numpy as np
from nltk.corpus import stopwords as nltk_stopwords
from sklearn.model_selection import train_test_split

import preprocessor as pr
import spellcorrector
import vectorization as vect
import my_models
import generator as gen
import math
import utils

from keras.callbacks import ModelCheckpoint
from keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from utils import tweet_len
from sklearn.metrics import classification_report, confusion_matrix

nltk.download('stopwords')
nltk.download('punkt')

pd.set_option('display.max_columns', None)
# -

# Create the objects

preprocessor = pr.Preprocessor(stopwords=nltk_stopwords.words('english'))

vec_size = 200

vectorizer = vect.WordEmbeddingsVectorizer(embeddings_file='../MyData/glove.twitter.27B.200d.txt', vec_size=vec_size)

# Read the data

data = pd.read_csv("../MyData/twitter_hate_data.csv")
print("Data shape:", data.shape, ", data columns:", data.columns)

# #### Split the data to train and test set respectively

train_data, test_data, train_labels, test_labels = train_test_split(data["tweet"], data["class"], test_size=0.1,
                                                                    random_state=0)

# #### Preprocess train data and save it to a csv file

preprocessed_train_data = pd.DataFrame({'tweet': preprocessor.preprocess(train_data)})
preprocessed_train_data["class"] = train_labels
preprocessed_train_data["length"] = preprocessed_train_data.apply(lambda row: tweet_len(row), axis=1)
preprocessed_train_data.sort_values(by=['length'], inplace=True, ignore_index=True)
preprocessed_train_data.to_csv("../MyData/twitter_preprocessed_train_data.csv", header=False)

# #### Preprocess test data and save it to a csv file

preprocessed_test_data = pd.DataFrame({'tweet': preprocessor.preprocess(test_data)})
preprocessed_test_data["class"] = test_labels
preprocessed_test_data["length"] = preprocessed_test_data.apply(lambda row: tweet_len(row), axis=1)
preprocessed_test_data.sort_values(by=['length'], inplace=True, ignore_index=True)
preprocessed_test_data.to_csv("../MyData/twitter_preprocessed_test_data.csv", header=False)

# ##### Read from csv so we won't have to preprocess text each time

preprocessed_train_data = pd.read_csv("../MyData/twitter_preprocessed_train_data.csv", names=["index", "tweet", "class",
                                                                                              "length"])
preprocessed_test_data = pd.read_csv("../MyData/twitter_preprocessed_test_data.csv", names=["index", "tweet", "class",
                                                                                            "length"])

print("Train data shape:", preprocessed_train_data.shape, ", data columns:", preprocessed_train_data.columns)
print("Test data shape:", preprocessed_test_data.shape, ", data columns:", preprocessed_test_data.columns)

# Change the classes from a string to categorical representation

train_labels = preprocessed_train_data["class"]
train_classes = pd.Series(train_labels, dtype="category")
le = LabelEncoder()
train_categorical_classes = to_categorical(le.fit_transform(train_classes))

test_labels = preprocessed_test_data["class"]
test_classes = pd.Series(test_labels, dtype="category")
le = LabelEncoder()
test_categorical_classes = to_categorical(le.fit_transform(test_classes))

# Initialize all the neccessary variables

batch_size = 256
layers = 1
go_backwards = True
use_embeddings = False
epochs = 20
model_name = 'one_bi_glove_20ep_model'
target_names = ['hate', 'neutral']
num_classes = len(target_names)

train_size = len(preprocessed_train_data['tweet'])
test_size = len(preprocessed_test_data['tweet'])

# Create two generators that will generate the training and the testing dataset respectively

train_generator = gen.KerasBatchGenerator(preprocessed_train_data['tweet'], train_categorical_classes, train_size, batch_size, num_classes, vectorizer, vec_size)
test_generator = gen.KerasBatchGenerator(preprocessed_test_data['tweet'], test_categorical_classes, test_size, batch_size, num_classes, vectorizer, vec_size)

# Prepare the generators for the models with the embeddings layer and any other data that we will need

# +
import wordEmbeddingsGenerator as gen

# Initialize the variables
vec_size = 200
batch_size = 256
use_embeddings = True
target_names = ['hate', 'neutral']
num_classes = len(target_names)

# Make the word to id dictionary
train_whole_text = utils.make_whole_text(preprocessed_train_data["tweet"])
test_whole_text = utils.make_whole_text(preprocessed_test_data["tweet"])

whole_text = train_whole_text + ' ' + test_whole_text

word_list = whole_text.split()
word_to_id = utils.build_vocab(word_list)

vocab_size = len(word_to_id)

# Get the length of the longest sequence
lengths = [len(x) for x in preprocessed_train_data["tweet"]]
lengths = lengths + [len(x) for x in preprocessed_test_data["tweet"]]
maxlen = max(lengths)

# Make the generator
# 2 is the number of classes
train_generator = gen.wordEmbeddingsGenerator(preprocessed_train_data["tweet"],
                                              train_categorical_classes,
                                              len(preprocessed_train_data),
                                              batch_size, 2, True, 
                                              vocab_size=vocab_size,
                                              word_to_id=word_to_id,
                                              maxlen=maxlen)

test_generator = gen.wordEmbeddingsGenerator(preprocessed_test_data["tweet"],
                                              test_categorical_classes,
                                              len(preprocessed_test_data),
                                              batch_size, 2, True, 
                                              vocab_size=vocab_size,
                                              word_to_id=word_to_id,
                                              maxlen=maxlen)

# +
epochs = 4
layers = 2
go_backwards = True
model_name = 'two_bi_layer_15ep_model'
model = my_models.make_model(layers=layers, go_backwards=go_backwards, use_embeddings=use_embeddings, 
                             batch_size=batch_size, vec_size=vec_size, vocab_size=vocab_size, input_length=maxlen)
model.load_weights('../MyModels/two_bi_layer_15ep_model-11.hdf5')

x_train = preprocessed_train_data['tweet']
y_train = train_categorical_classes

x_test = preprocessed_test_data['tweet']
y_test = test_categorical_classes


train_size = len(x_train)
test_size = len(x_test)

print("Evaluating the model")
result = model.evaluate(test_generator.generate(), steps=math.ceil(test_size / batch_size), batch_size=batch_size)

# Because our test data set does not divide exactly the batch size we will
# extend it with data from the beggining so it can fit exactly the batch
# size. We do this in order to make predictions for all the data. We will 
# then remove the extra data and make our reports with the initial test data
test_data_tweets = []

if use_embeddings:
    for tweet in x_test:
        test_data_tweets.append(rw.words_to_ids(tweet, word_to_id, maxlen))

    for i in range(test_size, math.ceil(test_size / batch_size)*batch_size):
        tweet = x_test.iloc[i - test_size]
        test_data_tweets.append(rw.words_to_ids(tweet, word_to_id, maxlen))
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
rw.print_report(result, y_true, y_pred, target_names, '../MyReports/' + model_name + '_report.txt')

# +
layers = 2
epochs = 5
go_backwards = False
model_name = 'two_uni_layer_15ep_model'

model = my_models.make_model(layers=layers, go_backwards=go_backwards, use_embeddings=use_embeddings, 
                             batch_size=batch_size, vec_size=vec_size, vocab_size=vocab_size, input_length=maxlen)
model.load_weights('../MyModels/two_uni_layer_15ep_model-10.hdf5')

x_train = preprocessed_train_data['tweet']
y_train = train_categorical_classes

x_test = preprocessed_test_data['tweet']
y_test = test_categorical_classes


train_size = len(x_train)
test_size = len(x_test)


print("Evaluating the model")
result = model.evaluate(test_generator.generate(), steps=math.ceil(test_size / batch_size), batch_size=batch_size)

# Because our test data set does not divide exactly the batch size we will
# extend it with data from the beggining so it can fit exactly the batch
# size. We do this in order to make predictions for all the data. We will 
# then remove the extra data and make our reports with the initial test data
test_data_tweets = []

if use_embeddings:
    for tweet in x_test:
        test_data_tweets.append(rw.words_to_ids(tweet, word_to_id, maxlen))

    for i in range(test_size, math.ceil(test_size / batch_size)*batch_size):
        tweet = x_test.iloc[i - test_size]
        test_data_tweets.append(rw.words_to_ids(tweet, word_to_id, maxlen))
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
rw.print_report(result, y_true, y_pred, target_names, '../MyReports/' + model_name + '_report.txt')
