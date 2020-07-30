from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tensorflow.keras.layers import Input, Dense, Dropout, Embedding, Flatten, Reshape, Bidirectional, LSTM, Concatenate
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

# LSTM model
input_x_LSTM= Input(batch_shape =(None, max_length))
Embedding_LSTM= model1.layers[2](input_x_LSTM) # 미리 학습된 model1의 embedding layer를 불러와 그대로 쓰기
biLSTM_LSTM= Bidirectional(LSTM(64))(Embedding_LSTM)
Output_LSTM= Dense(1, activation= 'sigmoid')(biLSTM_LSTM)

model_LSTM= Model(input_x_LSTM, Output_LSTM)
model_LSTM.layers[1].trainable= True

model_LSTM.compile(loss= 'binary_crossentropy', optimizer= 'adam')
hist_LSTM= model_LSTM.fit(x_train, y_train, validation_data= [x_test, y_test],
               batch_size= 100, epochs= 100, callbacks= [earlystopping])

# LSTM Loss
plt.plot(hist_LSTM.history['loss'], label= 'Train loss')
plt.plot(hist_LSTM.history['val_loss'], label= 'Test loss')
plt.legend()
plt.title('Loss history')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.show()
