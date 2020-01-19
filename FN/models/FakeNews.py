import numpy as np
import pandas as pd


import numpy as np
import pandas as pd

df = pd.read_csv('./fulldata.csv')
print(df)

train_data = df[['title', 'text']].values.tolist()
train_labels = df['label'].values.tolist()

train_data = np.asarray(train_data).tolist()
train_labels = np.asarray(train_labels).tolist()

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding
from tensorflow.keras.utils import to_categorical
import pickle

MAX_NB_WORDS=50000 #dictionary size
MAX_SEQUENCE_LENGTH=1500 #max word length of each individual article
EMBEDDING_DIM=300 #dimensionality of the embedding vector (50, 100, 200, 300)
tokenizer = Tokenizer(num_words=MAX_NB_WORDS, filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~')

def tokenize_trainingdata(texts, labels):
    tokenizer.fit_on_texts(texts)
    pickle.dump(tokenizer, open('./tokenizer.p', 'wb'))

    sequences = tokenizer.texts_to_sequences(texts)

    word_index = tokenizer.word_index
    print('Found %s unique tokens.' % len(word_index))

    data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)

    labels = to_categorical(labels, num_classes=len(set(labels)))

    return data, labels, word_index

#and run it
X, Y, word_index = tokenize_trainingdata(train_data, train_labels)

train_data = X[:int(len(X)*0.9)]
train_labels = Y[:int(len(X)*0.9)]
test_data = X[int(len(X)*0.9):int(len(X)*0.95)]
test_labels = Y[int(len(X)*0.9):int(len(X)*0.95)]
valid_data = X[int(len(X)*0.95):]
valid_labels = Y[int(len(X)*0.95):]

def load_embeddings(word_index, embeddingsfile='./glove.6B.%id.txt' %EMBEDDING_DIM):
    embeddings_index = {}
    f = open(embeddingsfile, 'r', encoding='utf8')
    for line in f:
        #here we parse the data from the file
        values = line.split(' ') #split the line by spaces
        word = values[0] #each line starts with the word
        coefs = np.asarray(values[1:], dtype='float32') #the rest of the line is the vector
        embeddings_index[word] = coefs #put into embedding dictionary
    f.close()

    print('Found %s word vectors.' % len(embeddings_index))

    embedding_matrix = np.zeros((len(word_index) + 1, EMBEDDING_DIM))
    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector
    
    embedding_layer = Embedding(len(word_index) + 1,
                                EMBEDDING_DIM,
                                weights=[embedding_matrix],
                                input_length=MAX_SEQUENCE_LENGTH,
                                trainable=False)
    return embedding_layer
    
#and build the embedding layer
embedding_layer = load_embeddings(word_index)


from tensorflow.keras import Sequential, Model, Input
from tensorflow.keras.layers import Conv1D, MaxPooling1D, AveragePooling1D, Flatten, Dense, GlobalAveragePooling1D, Dropout, LSTM, RNN, SimpleRNN, Conv2D, GlobalMaxPooling1D
from tensorflow.keras import callbacks
#CuDNNLSTM
def LSTM_model(sequence_input, embedded_sequences, classes=2):
    x = LSTM(32,
                  return_sequences=True)(embedded_sequences)
    x = LSTM(64,
                  return_sequences=True)(x)
    x = LSTM(128)(x)
    x = Dense(4096,
              activation='relu')(x)
    x = Dense(1024,
              activation='relu')(x)
    preds = Dense(classes,
              activation='softmax')(x)

    model = Model(sequence_input, preds)
    return model

