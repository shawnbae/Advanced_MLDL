from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt
import nltk
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import string
from nltk import pos_tag
from nltk.stem import PorterStemmer
import collections
from tensorflow.keras.layers import Input, Dense, Dropout, Dot, Embedding, Flatten, Reshape
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping

def preprocessing(text):
    text2 = "".join([" " if ch in string.punctuation else ch for ch in text])
    tokens = nltk.word_tokenize(text2)
    tokens = [word.lower() for word in tokens]
    
    stopwds = stopwords.words('english')
    tokens = [token for token in tokens if token not in stopwds]
    
    tokens = [word for word in tokens if len(word)>=3]
    
    stemmer = PorterStemmer()
    tokens = [stemmer.stem(word) for word in tokens]

    tagged_corpus = pos_tag(tokens)    
    
    Noun_tags = ['NN','NNP','NNPS','NNS']
    Verb_tags = ['VB','VBD','VBG','VBN','VBP','VBZ']

    lemmatizer = WordNetLemmatizer()

    def prat_lemmatize(token,tag):
        if tag in Noun_tags:
            return lemmatizer.lemmatize(token,'n')
        elif tag in Verb_tags:
            return lemmatizer.lemmatize(token,'v')
        else:
            return lemmatizer.lemmatize(token,'n')
    
    pre_proc_text =  " ".join([prat_lemmatize(token,tag) for token,tag in tagged_corpus])             

    return pre_proc_text

lines = []
fin = open("./dataset/alice_in_wonderland.txt", "r")
for line in fin:
    if len(line) == 0:
        continue
    lines.append(preprocessing(line))
fin.close()


word2idx = {w:(i+1) for i,(w,_) in enumerate(counter.most_common())}
idx2word = {v:k for k,v in word2idx.items()}


xs = []  
ys = []
label= [] 
for line in lines:
    embedding = [word2idx[w.lower()] for w in nltk.word_tokenize(line)]

    triples = list(nltk.trigrams(embedding))
    
    w_lefts = [x[0] for x in triples]
    w_centers = [x[1] for x in triples]
    w_rights = [x[2] for x in triples]
    
    xs.extend(w_centers)
    xs.extend(w_centers)
    label.extend([1 for i in range(len(w_centers))])
        
    ys.extend(w_lefts)
    ys.extend(w_rights)
    label.extend([0 for i in range(len(w_centers))])

vocab_size = len(word2idx) + 1  # 사전의 크기

ohe = OneHotEncoder(categories = [range(vocab_size)])
X = ohe.fit_transform(np.array(xs).reshape(-1, 1)).todense()
Y= np.array(label).reshape(-1,1)

Xtrain, Xtest, Ytrain, Ytest, xstr, xsts = train_test_split(X, Y, xs, test_size=0.2)
print(Xtrain.shape, Xtest.shape, Ytrain.shape, Ytest.shape)

# 딥러닝 모델 생성
BATCH_SIZE= 128
NUM_EPOCHS= 20
max_features= vocab_size
embedding_dim= 64

input_word_layer= Input(shape= (vocab_size,), name= "Input_words")
Word_Embed= Dense(embedding_dim, activation= 'relu')(input_word_layer)
Word_Embed_reshaped= Reshape((embedding_dim, 1))(Word_Embed)

input_target_layer= Input(shape= (vocab_size,), name= "Input_targets")
Context_Embed= Dense(embedding_dim, activation= 'relu')(input_target_layer)
Context_Embed_reshaped= Reshape((embedding_dim, 1))(Context_Embed)

Dotted= Dot(axes= 1)([Word_Embed_reshaped, Context_Embed_reshaped])
Output= Dense(1, activation= 'sigmoid')(Dotted)
Output_reshaped= Reshape((1,))(Output)
model= Model([input_word_layer,input_target_layer], Output_reshaped)
model.compile(loss= 'binary_crossentropy', optimizer= 'rmsprop')

es= EarlyStopping(monitor= 'val_loss', patience= 10, mode= 'min')

hist= model.fit([Xtrain, Xtrain], Ytrain, validation_data= ([Xtest,Xtest],Ytest),
                batch_size= BATCH_SIZE, epochs=100, callbacks= [es])




