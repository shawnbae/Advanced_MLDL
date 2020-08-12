from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tensorflow.keras.layers import Input, Dense, Dropout, Embedding, Flatten, Reshape, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.datasets import imdb
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import load_model

# IMDB load
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words= max_features)

x_train= sequence.pad_sequences(x_train, maxlen= max_length)
x_test= sequence.pad_sequences(x_test, maxlen= max_length)

# skipgram model load
model1= load_model('skipgram_model.h5')

# earlystopping callback
earlystopping= EarlyStopping(patience= 5)

# LSTM model with random initiation
max_features= 6000
max_length= 400
embedding_dim= 100

x_input= Input(batch_shape= (None, max_length))
WordEmbeddingLayer= Embedding(max_features, embedding_dim)(x_input)
x_LSTM= Bidirectional(LSTM(64))(WordEmbeddingLayer)
x_Output= Dense(1, activation= 'sigmoid')(x_LSTM)

model_LSTM_R= Model(x_input, x_Output)
model_LSTM_R.compile(loss= 'binary_crossentropy', optimizer= 'adam')

hist_LSTM_R= model_LSTM_R.fit(x_train, y_train, validation_data= [x_test, y_test],
                              batch_size= 100, epochs= 100, callbacks= [earlystopping])

