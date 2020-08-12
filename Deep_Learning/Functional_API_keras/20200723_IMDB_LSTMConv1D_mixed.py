from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, Activation
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import Conv1D, GlobalMaxPooling1D
from tensorflow.keras.layers import Bidirectional, LSTM, Input, Flatten, concatenate
from tensorflow.keras.datasets import imdb
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import numpy as np

max_features= 6000
max_length= 400

batch_size = 32
embedding_dims = 60
num_kernels = 260
kernel_size = 3
hidden_dims = 300
epochs = 10

(x_train, y_train), (x_test, y_test)= imdb.load_data(num_words= max_features)

x_train = sequence.pad_sequences(x_train, maxlen=max_length)
x_test = sequence.pad_sequences(x_test, maxlen=max_length)

# 병렬 Embedding구조
xInput = Input(batch_shape=(None, max_length))
xEmbed1 = Embedding(max_features, embedding_dims, input_length= max_length)(xInput) # CNN
xEmbed2 = Embedding(max_features, 60)(xInput) # LSTM

xConv= Conv1D(num_kernels, kernel_size, padding= 'valid', activation= 'relu', strides=1)(xEmbed1)
xGlobal= GlobalMaxPooling1D()(xConv)
xHidden_CNN = Dense(hidden_dims)(xGlobal)
xActivation= Activation('relu')(xHidden_CNN)
Flat_CNN= Flatten()(xActivation)

xLstm = Bidirectional(LSTM(64))(xEmbed2)
xHidden_LSTM = Dense(hidden_dims)(xLstm)
Flat_LSTM= Flatten()(xHidden_LSTM)

xOutput= Dense(1, activation= 'sigmoid')(concatenate([Flat_CNN, Flat_LSTM]))
model = Model(xInput, xOutput)
model.compile(loss='binary_crossentropy', optimizer='adam')

hist = model.fit(x_train, y_train, 
                 batch_size=32, 
                 epochs=1,
                 validation_data = (x_test, y_test))

# 직렬 Embedding구조
xInput= Input(batch_shape= (None, max_length))
xEmbed= Embedding(max_features, embedding_dims, input_length= max_length)(xInput)

xConv= Conv1D(num_kernels, kernel_size, padding= 'valid', activation= 'relu', strides=1)(xEmbed)
xGlobal= GlobalMaxPooling1D()(xConv)
xHidden_CNN = Dense(hidden_dims)(xGlobal)
xActivation= Activation('relu')(xHidden_CNN)
Flat_CNN= Flatten()(xActivation)

xLstm = Bidirectional(LSTM(64))(xEmbed)
xHidden_LSTM = Dense(hidden_dims)(xLstm)
Flat_LSTM= Flatten()(xHidden_LSTM)

xOutput= Dense(1, activation= 'sigmoid')(concatenate([Flat_CNN, Flat_LSTM]))
model2= Model(xInput, xOutput)
model2.compile(loss= 'binary_crossentropy', optimizer= 'adam')

hist2= model2.fit(x_train, y_train,
                  batch_size= 32,
                  epochs=1,
                  validation_data= (x_test, y_test))

# 병렬 Embedding History graph
plt.plot(hist.history['loss'], label='Train loss')
plt.plot(hist.history['val_loss'], label = 'Test loss')
plt.legend()
plt.title("Loss history")
plt.xlabel("epoch")
plt.ylabel("loss")
plt.show()

# 직렬 Embedding History graph
plt.plot(hist2.history['loss'], label='Train loss')
plt.plot(hist2.history['val_loss'], label = 'Test loss')
plt.legend()
plt.title("Loss history")
plt.xlabel("epoch")
plt.ylabel("loss")
plt.show()

# 병렬 predicting
y_train_predprob = model.predict(x_train, batch_size=batch_size)
y_test_predprob = model.predict(x_test, batch_size=batch_size)

y_train_predclass= np.where(y_train_predprob> 0.5, 1, 0)
y_test_predclass= np.where(y_test_predprob > 0.5, 1, 0)

y_train_predclass.shape = y_train.shape
y_test_predclass.shape = y_test.shape

print (("Train accuracy:"),(np.round(accuracy_score(y_train,y_train_predclass),3)))  
print (("Test accuracy:"),(np.round(accuracy_score(y_test,y_test_predclass),3)))     

# 직렬 predicting
y_train_predprob2 = model2.predict(x_train, batch_size=batch_size)
y_test_predprob2 = model2.predict(x_test, batch_size=batch_size)

y_train_predclass2= np.where(y_train_predprob2> 0.5, 1, 0)
y_test_predclass2= np.where(y_test_predprob2 > 0.5, 1, 0)

y_train_predclass2.shape = y_train.shape
y_test_predclass2.shape = y_test.shape

print (("Train accuracy:"),(np.round(accuracy_score(y_train,y_train_predclass2),3)))  
print (("Test accuracy:"),(np.round(accuracy_score(y_test,y_test_predclass2),3)))     
