from tensorflow.keras.layers import Input,Dense,Dropout,Reshape
from tensorflow.keras.models import Model
from sklearn.model_selection import train_test_split
import nltk
import numpy as np
import string
import random

with open("./dataset/alice_in_wonderland.txt", 'r') as content_file:
    content = content_file.read()

content2 = " ".join("".join([" " if ch in string.punctuation else ch for ch in content]).split())
 
tokens = nltk.word_tokenize(content2)
tokens = [word.lower() for word in tokens if len(word)>=2]

N = 5
quads = list(nltk.ngrams(tokens,N))

newl_app = []
for ln in quads:
    newl = " ".join(ln)        
    newl_app.append(newl)

from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer()

x_pentgm = []
y_pentgm = []

for l in newl_app:
    x_str = " ".join(l.split()[0:N-2])
    y_str = " ".join(l.split()[N-2:])   
    x_pentgm.append(x_str)
    y_pentgm.append(y_str)

x_pentgm_check = vectorizer.fit_transform(x_pentgm).todense()
y_pentgm_check = vectorizer.fit_transform(y_pentgm).todense()

dictnry= vectorizer.vocabulary_
rev_dictnry = {v:k for k,v in dictnry.items()}

X = np.array(x_pentgm_check)
Y = np.array(y_pentgm_check)

Xtrain, Xtest, Ytrain, Ytest,xtrain_tg,xtest_tg = train_test_split(X, Y,x_pentgm, test_size=0.3,random_state=42)

# Modeling
BATCH_SIZE = 128
NUM_EPOCHS = 100

input_layer = Input(shape = (Xtrain.shape[1],),name="input")

first_layer = Dense(1000,activation='relu',name = "first")(input_layer)
first_dropout = Dropout(0.5,name="firstdout")(first_layer)

second_layer = Dense(800,activation='relu',name="second")(first_dropout)

third_layer = Dense(1000,activation='relu',name="third")(second_layer)
third_dropout = Dropout(0.5,name="thirdout")(third_layer)

fourth_layer = Dense(Ytrain.shape[1],activation='sigmoid',name = "fourth")(third_dropout)

history = Model(input_layer,fourth_layer)
history.compile(optimizer = "adam",loss="binary_crossentropy",metrics=["accuracy"])

history.fit(Xtrain, Ytrain, batch_size=BATCH_SIZE,epochs=10, verbose=1,validation_split = 0.2)

# prediction
Y_pred= history.predict(Xtest)

for i in range(100):
    print (i,xtest_tg[i],"|",rev_dictnry[np.argmax(Ytest[i])],"|",rev_dictnry[np.argmax(Y_pred[i])], rev_dictnry[np.argmax(Y_pred[i])-1])

NUM_DISPLAY = 10
for i in random.sample(range(len(xtest_tg)), NUM_DISPLAY):
	print (i,xtest_tg[i],"|",rev_dictnry[np.argmax(Ytest[i])],"|",rev_dictnry[np.argmax(Y_pred[i])])

