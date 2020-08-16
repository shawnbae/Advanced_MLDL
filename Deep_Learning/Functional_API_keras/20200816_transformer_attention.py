import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import Model
import pickle
import keras
from keras_multi_head import MultiHead
from keras_multi_head import MultiHeadAttention


# 채팅 dataset upload
DATA_PATH = './dataset/6-1.ChatBotData.csv'
def load_data():
    data_df = pd.read_csv(DATA_PATH, header=0)
    question, answer = list(data_df['Q']), list(data_df['A'])
    
    train_input, eval_input, train_label, eval_label = \
        train_test_split(question, answer, test_size=0.1, random_state=42)
        
    return train_input, train_label, eval_input, eval_label

train_input, train_label, eval_input, eval_label = load_data()

# Seq2seq을 위해 정제했던 dataset upload
with open('./dataset/6-1.vocabulary.pickle', 'rb') as f:
    word2idx,  idx2word = pickle.load(f)
    
# 학습 데이터 : 인코딩, 디코딩 입력, 디코딩 출력을 읽어온다.
with open('./dataset/6-1.train_data.pickle', 'rb') as f:
    trainXE, trainXD, trainYD = pickle.load(f)
	
# 평가 데이터 : 인코딩, 디코딩 입력, 디코딩 출력을 만든다.
with open('./dataset/6-1.eval_data.pickle', 'rb') as f:
    testXE, testXD, testYD = pickle.load(f)
    
# Hyperparameters
vocab_size = len(idx2word)
maxlen= 10
embed_dim = 128
num_heads = 8
ff_dim = 128

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

# TokenAndPositionEmbedding Layer Class
class TokenAndPositionEmbedding(layers.Layer):
    def __init__(self, maxlen, vocab_size, embed_dim):
        super(TokenAndPositionEmbedding, self).__init__()
        self.token_emb = layers.Embedding(input_dim=vocab_size, output_dim=embed_dim)
        self.pos_emb = layers.Embedding(input_dim=maxlen, output_dim=embed_dim)

    def call(self, x):
        maxlen = tf.shape(x)[-1]
        positions = tf.range(start=0, limit=maxlen, delta=1)
        positions = self.pos_emb(positions)
        x = self.token_emb(x)
        return x + positions

# Encoder 
inputs_encoder= layers.Input(shape= (None,))
embed_encoder= TokenAndPositionEmbedding(maxlen, vocab_size, embed_dim)(inputs_encoder)

query_encoder_1= layers.Dense(embed_dim//num_heads,activation= 'linear')(embed_encoder)
key_encoder_1= layers.Dense(embed_dim//num_heads, activation= 'linear')(embed_encoder)
value_encoder_1= layers.Dense(embed_dim//num_heads, activation= 'linear')(embed_encoder)

QKV_concat= layers.Concatenate()([query_encoder_1, key_encoder_1, value_encoder_1])

Multihead_encoder= MultiHeadAttention(head_num= num_heads)(embed_encoder, mask=None)

Add1_encoder= layers.Add()([Multihead_encoder, layers.Concatenate(axis=1)([embed_encoder_query, embed_encoder_key, embed_encoder_value])])
Norm1_encoder= layers.LayerNormalization()(Add1_encoder)
FFN_encoder= layers.Dense(ff_dim, activation= 'relu')(Norm1_encoder)
Add2_encoder= layers.Add()([FFN_encoder,Norm1_encoder])
output_encoder= layers.LayerNormalization()(Add2_encoder)

model_encoder= Model(inputs_encoder, output_encoder)


# Decoder
inputs_decoder= layers.Input(shape= (maxlen,))
embed_decoder= TokenAndPositionEmbedding(maxlen, vocab_size, embed_dim)(inputs_decoder)
Multihead_decoder= MultiHeadAttention(head_num= num_heads)(embed_encoder, mask= ) 



model = keras.Model(inputs=inputs, outputs=outputs)

