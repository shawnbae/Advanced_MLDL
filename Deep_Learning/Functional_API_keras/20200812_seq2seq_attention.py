import tensorflow as tf
from tensorflow.keras.layers import Input, LSTM, Dense
from tensorflow.keras.layers import Embedding, TimeDistributed
from tensorflow.keras.models import Model
import tensorflow.keras.backend as K
import numpy as np

# Hyperparameter set
VOCAB_SIZE= len(idx2word)
EMB_SIZE= 128
LSTM_HIDDEN= 128
MAX_SEQUENCE_LEN= 10

# Common Embedding Layer
wordEmbedding= Embedding(input_dim= VOCAB_SIZE, output_dim= EMB_SIZE)

# Encoder Layer
class Encoder(Model):
  def __init__(self):
    super(Encoder, self).__init__()
    self.encEMB= wordEmbedding
    self.lstm1= LSTM(LSTM_HIDDEN, return_sequences=True, return_state=True)
    self.lstm2= LSTM(LSTM_HIDDEN, return_state= True)

  def call(self, input_encoder_X, training=False, mask=None):
    x = self.encEMB(input_encoder_X)
    ey1, eh1, ec1 = self.lstm1(input_encoder_X)
    _, eh2, ec2 = self.lstm2(ey1)
    
    return eh2, ec2

# Decoder Layer
class Decoder(Model):
  def __init__(self):
    super(Decoder, self).__init__()
    self.decEMB = wordEmbedding
    self.lstm1 = LSTM(LSTM_HIDDEN, return_sequences=True, return_state=True)
    self.lstm2 = LSTM(LSTM_HIDDEN, return_sequences=True, return_state=True)
    self.dense = TimeDistributed(Dense(VOCAB_SIZE, activation='softmax'))

  def call(self, input_decoder_X, training=False, mask=None):
    x, h, c = input_decoder_X
    x = self.decEMB(x)
    dx1, dh1, dc1 = self.lstm(x, initial_state=[h, c])
    return self.dense(x), h, c

class Seq2seq(Model):
  def __init__(self, sos, eos):
    super(Seq2seq, self).__init__()
    self.enc = Encoder()
    self.dec = Decoder()
    self.sos = sos
    self.eos = eos

  def call(self, inputs, training=False, mask=None):
    if training is True:
      x, y = inputs
      h, c = self.enc(x)
      y, _, _ = self.dec((y, h, c))
      return y

    else:
      x = inputs
      h, c = self.enc(x)

      y = tf.convert_to_tensor(self.sos)
      y = tf.reshape(y, (1, 1))

      seq = tf.TensorArray(tf.int32, 64)

      for idx in tf.range(64):
        y, h, c = self.dec([y, h, c])
        y = tf.cast(tf.argmax(y, axis=-1), dtype=tf.int32)
        y = tf.reshape(y, (1, 1))
        seq = seq.write(idx, y)

        if y == self.eos:
          break 
      return tf.reshape(seq.stack(), (1, 64))
