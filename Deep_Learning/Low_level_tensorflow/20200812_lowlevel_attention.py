import tensorflow as tf
from tensorflow.keras.layers import Input, LSTM, Dense, Reshape, Dot, Flatten, Concatenate, Activation
from tensorflow.keras.layers import Embedding, TimeDistributed
from tensorflow.keras.layers import Attention, AdditiveAttention
from tensorflow.keras.models import Model
import tensorflow.keras.backend as K
import numpy as np

# wordEmbedding Layer
K.clear_session()
wordEmbedding = Embedding(input_dim=VOCAB_SIZE, output_dim=EMB_SIZE)

# Encoder
encoderX = Input(batch_shape=(None, MAX_SEQUENCE_LEN))
encEMB = wordEmbedding(encoderX)
encLSTM1 = LSTM(LSTM_HIDDEN, return_sequences=True, return_state = True)
encLSTM2 = LSTM(LSTM_HIDDEN, return_sequences=True, return_state = True)
ey1, eh1, ec1 = encLSTM1(encEMB)    # LSTM 1층 
ey2, eh2, ec2 = encLSTM2(ey1)         # LSTM 2층

# Decoder
decoderX = Input(batch_shape=(None, MAX_SEQUENCE_LEN))
decEMB = wordEmbedding(decoderX)
decLSTM1 = LSTM(LSTM_HIDDEN, return_sequences=True, return_state=True)
decLSTM2 = LSTM(LSTM_HIDDEN, return_sequences=True, return_state=True)
dy1, _, _ = decLSTM1(decEMB, initial_state = [eh1, ec1])
dy2, _, _ = decLSTM2(dy1, initial_state = [eh2, ec2])
decOutput = TimeDistributed(Dense(VOCAB_SIZE, activation='softmax'))
outputY = decOutput(dy2)

# Low-level Attention
attentionprocess_1= Dot(axes=2)([dy2,ey2])
attentionprocess_2= Activation('softmax')(attentionprocess_1)
attentionprocess_3= Dot(axes=1)([attentionprocess_2, ey2])
attentionprocess_5= Concatenate(axis=2)([attentionprocess_3,dy2])
