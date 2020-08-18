import re
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import sentencepiece as spm
from keras_transformer import get_model
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
from tensorflow.python.keras.preprocessing.text import Tokenizer
import matplotlib.pyplot as plt

MOVIE_REVIEW_DATA = '4-8.ratings_review.npy'
MOVIE_LABEL_DATA = '4-8.ratings_label.npy'
DATA_IN_PATH = './dataset/'

# 전처리 전의 데이터를 가져온다.
DATA_IN_PATH = 'c:/Users/soohan/MultiCampus/python_nlp2/dataset/'
movie_data = pd.read_csv(DATA_IN_PATH + '4-8.ratings.txt', 
                         header = 0, delimiter = '\t', quoting = 3)
movie_data = movie_data.dropna() # 공백 제거

# punctuation 제거
FILTERS = "([~.,!?\"':;)(])^*"

# 전처리 작업
def preprocessing(review):
    review_text = re.sub("[^가-힣ㄱ-ㅎㅏ-ㅣ]", "", review)
    return review_text

clean_review= []
for i, review in enumerate(movie_data['document']):
    # 비어있는 데이터에서 멈추지 않도록 string인 경우만 진행
    if type(review) == str:
        clean_review.append(preprocessing(review))
        
    if i % 100 == 0:
        print('%d : %.2f%% 완료됨.' % (i, 100 * i / len(movie_data)))

# 전처리된 데이터를 따로 저장하여 import할 수 있게 한다.
data_file = "./dataset/naver_movie_review.txt"

f = open(data_file, 'w', encoding= "utf-8")
for sent in clean_review:
    f.write(sent + '\n')

# SentencePiece
templates= "--input={} \
            --pad_id=0 --pad_piece=<PAD>\
            --unk_id=1 --unk_piece=<UNK>\
            --bos_id=2 --bos_piece=<START>\
            --eos_id=3 --eos_piece=<END>\
            --model_prefix={} \
            --vocab_size={} \
            --character_coverage=1.0 \
            --model_type=unigram"
VOCAB_SIZE = 9000
model_prefix = "./dataset/naver_model"
params = templates.format(data_file, model_prefix, VOCAB_SIZE)

spm.SentencePieceTrainer.Train(params)
sp = spm.SentencePieceProcessor()
sp.Load(model_prefix + '.model')

with open(model_prefix + '.vocab', encoding='utf-8') as f:
    vocab = [doc.strip().split('\t') for doc in f]

word2idx = {k:v for v, [k, _] in enumerate(vocab)}
idx2word = {v:k for v, [k, _] in enumerate(vocab)}

# TransformerBlock
class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerBlock, self).__init__()
        self.att = MultiHeadSelfAttention(embed_dim, num_heads)
        self.ffn = keras.Sequential(
            [layers.Dense(ff_dim, activation="relu"), layers.Dense(embed_dim),]
        )
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs, training):
        attn_output = self.att(inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

# MultiHeadSelfAttention
class MultiHeadSelfAttention(layers.Layer):
    def __init__(self, embed_dim, num_heads=8):
        super(MultiHeadSelfAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embedding dimension = {embed_dim} should be divisible by number of heads = {num_heads}"
            )
        self.projection_dim = embed_dim // num_heads
        self.query_dense = layers.Dense(embed_dim)
        self.key_dense = layers.Dense(embed_dim)
        self.value_dense = layers.Dense(embed_dim)
        self.combine_heads = layers.Dense(embed_dim)

    def attention(self, query, key, value):
        score = tf.matmul(query, key, transpose_b=True)
        dim_key = tf.cast(tf.shape(key)[-1], tf.float32)
        scaled_score = score / tf.math.sqrt(dim_key)
        weights = tf.nn.softmax(scaled_score, axis=-1)
        output = tf.matmul(weights, value)
        return output, weights

    def separate_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.projection_dim))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, inputs):
        # x.shape = [batch_size, seq_len, embedding_dim]
        batch_size = tf.shape(inputs)[0]
        query = self.query_dense(inputs)  # (batch_size, seq_len, embed_dim)
        key = self.key_dense(inputs)  # (batch_size, seq_len, embed_dim)
        value = self.value_dense(inputs)  # (batch_size, seq_len, embed_dim)
        query = self.separate_heads(
            query, batch_size
        )  # (batch_size, num_heads, seq_len, projection_dim)
        key = self.separate_heads(
            key, batch_size
        )  # (batch_size, num_heads, seq_len, projection_dim)
        value = self.separate_heads(
            value, batch_size
        )  # (batch_size, num_heads, seq_len, projection_dim)
        attention, weights = self.attention(query, key, value)
        attention = tf.transpose(
            attention, perm=[0, 2, 1, 3]
        )  # (batch_size, seq_len, num_heads, projection_dim)
        concat_attention = tf.reshape(
            attention, (batch_size, -1, self.embed_dim)
        )  # (batch_size, seq_len, embed_dim)
        output = self.combine_heads(
            concat_attention
        )  # (batch_size, seq_len, embed_dim)
        return output
# dataset
MAX_LEN = 15
enc_input = []
for text, label in zip(clean_review, movie_label):
    enc_i = [word2idx[k] for k in sp.encode_as_pieces(text)]
    enc_input.append(enc_i)      

# input data
MAX_SEQUENCE_LENGTH = 8 # 문장 최대 길이
movie_input = pad_sequences(enc_input, maxlen=8, value = sp.pad_id(), padding='post', truncating='post')

movie_label = np.array(movie_data['label'])

# model
vocab_size= len(vocab)
embed_dim = 32  # Embedding size for each token
num_heads = 2  # Number of attention heads
ff_dim = 32  # Hidden layer size in feed forward network inside transformer

inputs = layers.Input(shape=(MAX_SEQUENCE_LENGTH,))
embedding_layer = layers.Embedding(input_dim= vocab_size, 
                                   output_dim= embed_dim, input_length= MAX_SEQUENCE_LENGTH)
x = embedding_layer(inputs)
transformer_block = TransformerBlock(embed_dim, num_heads, ff_dim)
x = transformer_block(x)
x = layers.GlobalAveragePooling1D()(x)
x = layers.Dropout(0.1)(x)
x = layers.Dense(20, activation="relu")(x)
x = layers.Dropout(0.1)(x)
outputs = layers.Dense(1, activation="sigmoid")(x)

model = keras.Model(inputs=inputs, outputs=outputs)
model.compile(loss= 'binary_crossentropy',optimizer= 'adam')

model.fit(movie_input, movie_label, epochs= 2, batch_size= 100)
