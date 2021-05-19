from keras.models import Sequential, load_model
from keras.layers import Dense, GaussianNoise, Activation, Embedding, Dropout, TimeDistributed, LSTM, Flatten, \
    GlobalAveragePooling1D, Masking


def make_model(vec_size=200, batch_size=2000, go_backwards=True, return_sequences=True, layers=2, dropout=0.2,
               rec_dropout=0.2, use_embeddings=False, vocab_size=500, input_length=1000):
    """
    Makes an LSTM neural network

    :param vec_size: The dimensionality of the input
    :param batch_size: The size of each batch that the dataset will split at
    :param go_backwards: Flag that determines if the network will be bi-directional or not
    :param return_sequences: Whether to return the last output. in the output sequence, or the full sequence
    :param layers: Number of LSTM layers
    :param dropout: Droupout rate for the drouput layers. If value is less than or 0 no dropout layer will be used
    :rec_dropout: Droupout rate for the LSTM layers
    :use_embeddings: If ser to true we will add a word embeddings layer, otherwise word embeddings will be provided
                     as the input to the network
    :vocab_size: Number of unique tokens
    :param input_length: Length of input sequences, when it is constant. Used when we have an embeddings layer

    Returns a keras model
    """

    model = Sequential()

    if layers < 1:
        raise ValueError("Please provide a positive integer for the number of layers")

    if use_embeddings:
        model.add(Embedding(vocab_size, vec_size, batch_size=batch_size, mask_zero=True, input_length=input_length))
        model.add(LSTM(vec_size, go_backwards=go_backwards, return_sequences=return_sequences, dropout=rec_dropout))
    else:
        model.add(LSTM(vec_size, batch_input_shape=(batch_size, None, vec_size), go_backwards=go_backwards,
                       return_sequences=return_sequences, dropout=rec_dropout))

    for i in range(1, layers):
        model.add(LSTM(vec_size, go_backwards=go_backwards, return_sequences=return_sequences, dropout=rec_dropout))

    if dropout > 0.0:
        model.add(Dropout(rate=dropout))

    model.add(GlobalAveragePooling1D())

    model.add(Dense(32, activation='relu'))

    model.add(Dense(2, activation='relu'))

    model.add(Activation('sigmoid'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['categorical_accuracy'])

    return model

