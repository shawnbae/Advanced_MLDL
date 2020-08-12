import tensorflow as tf
from tensorflow.keras.layers import Input, LSTM, Dense
from tensorflow.keras.layers import Embedding, TimeDistributed
from tensorflow.keras.models import Model
import tensorflow.keras.backend as K
import numpy as np

# dataset
print(word2idx)
print(idx2word)
print(trainXE)
print(trainXD)
print(trainYD)

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

      # LSTM으로 구현되었기 때문에 Hidden State와 Cell State를 출력으로 내준다.
      h, c = self.enc(x)

      # Hidden state와 cell state, shifted output을 초기값으로 입력 받고
      # 출력으로 나오는 y는 Decoder의 결과이기 때문에 전체 문장이 될 것이다.
      y, _, _ = self.dec((y, h, c))
      return y

    else:
      x = inputs
      h, c = self.enc(x)

      # Decoder 단에 제일 먼저 sos를 넣어주게끔 tensor화시키고
      y = tf.convert_to_tensor(self.sos)
      # shape을 맞춰주기 위한 작업이다.
      y = tf.reshape(y, (1, 1))

      # 최대 64길이 까지 출력으로 받을 것이다.
      seq = tf.TensorArray(tf.int32, 64)

      # tf.keras.Model에 의해서 call 함수는 auto graph모델로 변환이 되게 되는데,
      # 이때, tf.range를 사용해 for문이나 while문을 작성시 내부적으로 tf 함수로 되어있다면
      # 그 for문과 while문이 굉장히 효율적으로 된다.
      for idx in tf.range(64):
        y, h, c = self.dec([y, h, c])
        # 아래 두가지 작업은 test data를 예측하므로 처음 예측한값을 다시 다음 step의 입력으로 넣어주어야하기에 해야하는 작업이다.
        # 위의 출력으로 나온 y는 softmax를 지나서 나온 값이므로
        # 가장 높은 값의 index값을 tf.int32로 형변환해주고
        # 위에서 만들어 놓았던 TensorArray에 idx에 y를 추가해준다.
        y = tf.cast(tf.argmax(y, axis=-1), dtype=tf.int32)
        # 위의 값을 그대로 넣어주게 되면 Dimension이 하나밖에 없어서
        # 실제로 네트워크를 사용할 때 Batch를 고려해서 사용해야 하기 때문에 (1,1)으로 설정해 준다.
        y = tf.reshape(y, (1, 1))
        seq = seq.write(idx, y)

        if y == self.eos:
          break
      # stack은 그동안 TensorArray로 받은 값을 쌓아주는 작업을 한다.    
      return tf.reshape(seq.stack(), (1, 64))
