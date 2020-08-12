import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, LSTM, Embedding, Dropout, Reshape
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras import regularizers
import matplotlib.pyplot as plt

# Data import & split
TRAIN_INPUT_DATA = '4-1.train_input.npy'
TRAIN_LABEL_DATA = '4-1.train_label.npy'
DATA_IN_PATH = './dataset/'

input_data = np.load(open(DATA_IN_PATH + TRAIN_INPUT_DATA, 'rb'))
label_data = np.load(open(DATA_IN_PATH + TRAIN_LABEL_DATA, 'rb'))

TEST_SPLIT = 0.1
RANDOM_SEED = 13371447

train_input, test_input, train_label, test_label = train_test_split(input_data, label_data, 
                                                                    test_size=TEST_SPLIT, random_state=RANDOM_SEED)

# model build
vocab_size= 80000
embedding_dim= 60

xInput= Input(shape= (174,))
xembed= Embedding(vocab_size, embedding_dim)(xInput)
xLSTM= LSTM(64)(xembed)
xOutput= Dense(1, activation= 'sigmoid')(xLSTM)

model= Model(xInput, xOutput)
model.compile(loss= 'binary_crossentropy', optimizer= Adam(lr= 0.001))

# 학습
model.fit(train_input, train_label, epochs= 10, batch_size= 100, 
          validation_data= [test_input, test_label])

pred_label= model.predict(test_input)
plt.hist(pred_label)

predict_result= np.where(pred_label > 0.5, 1, 0)
predict_result.sum()

# Test accuracy
(predict_result == np.array(test_label).reshape(-1,1)).mean()
