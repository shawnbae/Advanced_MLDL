from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tensorflow.keras.layers import Input, Dense, Dropout, Embedding, Flatten, Reshape, Bidirectional, LSTM, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.datasets import imdb
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score

max_features= 6000
max_length= 400
embedding_dim= 100

(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words= max_features)

x_train= sequence.pad_sequences(x_train, maxlen= max_length)
x_test= sequence.pad_sequences(x_test, maxlen= max_length)

# skip-gram with negative sampling
input_x= Input(batch_shape= (None, max_length))
target_x= Input(batch_shape= (None, max_length))

Embed_input= Embedding(max_features, embedding_dim)(input_x)
Embed_target= Embedding(max_features, embedding_dim)(target_x)

Concat= Concatenate(axis= 1)([Embed_input, Embed_target])

Output_x= Dense(1, activation= 'sigmoid')(Concat)
Output_x_reshaped= Reshape((-1,1))(Output_x)
model_skipgram= Model([input_x, target_x], Output_x_reshaped)
model_skipgram.compile(loss= 'binary_crossentropy', optimizer = 'adam')

earlystopping= EarlyStopping(patience= 5) 
model_skipgram.fit([x_train, x_train], y_train, validation_data= ([x_test, x_test],y_test),
                   callbacks= [earlystopping], epochs= 100, batch_size= 100)

# save parameters
model_skipgram.save('c:/Users/soohan/MC_python_study/python_nlp/skipgram_model.h5')
