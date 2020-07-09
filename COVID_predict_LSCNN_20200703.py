import pandas as pd
import datetime
from datetime import datetime
from tensorflow.keras.layers import Dense, Input, LSTM, TimeDistributed,Bidirectional, concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam,Nadam
from sklearn.preprocessing import scale
import numpy as np
import matplotlib.pyplot as plt


data= pd.read_csv('dataset/covid_19_data.csv')
df_data= pd.DataFrame(data)
grouped_by_date= df_data.groupby('ObservationDate')

# 일자별 누적합
cum_confirmed_by_date= grouped_by_date.sum().Confirmed
cum_death_by_date= grouped_by_date.sum().Deaths
cum_recovered_by_date= grouped_by_date.sum().Recovered

# 일일합계 구하기
confirmed_list= []
for idx in range(len(cum_confirmed_by_date)):
    if idx==0:
        confirmed_list.append(cum_confirmed_by_date[0])
    else:
        confirmed_list.append(cum_confirmed_by_date[idx]-cum_confirmed_by_date[idx-1])

confirmed_by_date= pd.Series(confirmed_list, index= cum_confirmed_by_date.index)


death_list= []
for idx in range(len(cum_death_by_date)):
    if idx==0:
        death_list.append(cum_death_by_date[0])
    else:
        death_list.append(cum_death_by_date[idx]-cum_death_by_date[idx-1])

death_by_date= pd.Series(death_list, index= cum_death_by_date.index)

recovered_list= []
for idx in range(len(cum_death_by_date)):
    if idx==0:
        recovered_list.append(cum_recovered_by_date[0])
    else:
        recovered_list.append(cum_recovered_by_date[idx]-cum_recovered_by_date[idx-1])

recovered_by_date= pd.Series(recovered_list, index= cum_recovered_by_date.index)

by_date_df= pd.concat([confirmed_by_date,death_by_date,recovered_by_date],axis= 1)
by_date_df.columns= ['confirmed','death','recovered']
scaled_df= scale(by_date_df, axis=0)
#---------------------------------------------------------------------------#
# LSCNN
input_list= []
for i in range(nStep, len(by_date_df)):
    input_list.append(np.array(scaled_df[i-nStep:i]).reshape(-1,20,3))

output_list= []
for i in range(nStep+1, len(by_date_df)+1):
    output_list.append(np.array(scaled_df[i-nStep:i]).reshape(-1,20,3))
    
input_array_LSTM= np.array(input_list).reshape(-1,20,3)

input_list= []
for i in range(nStep, len(by_date_df)):
    input_list.append(np.array(scaled_df[i-nStep:i]).reshape(-1,20,3))

output_list= []
for i in range(nStep+1, len(by_date_df)+1):
    output_list.append(np.array(scaled_df[i-nStep:i]).reshape(-1,20,3))
    
input_array_CNN= np.array(input_list).reshape(-1,20,3)
output_array_2D= np.array(output_list).reshape(-1,3)[-141:]
test_array_LSCNN= np.array(output_list).reshape(-1,20,3)

# LSCNN modeling
nInput_LSTM = 3
nOutput_LSTM = 3
nStep_LSTM = 20
nHidden_LSTM = 50

nStep_CNN = 20
nFeature_CNN = 3

LSTM_x = Input(batch_shape=(None, nStep_LSTM, 3))
CNN_x = Input(batch_shape = (None, nStep_CNN, nFeature_CNN))

xLstm1 = LSTM(nHidden, return_sequences=True)(LSTM_x)
xLstm2 = Bidirectional(LSTM(nHidden), merge_mode='concat')(xLstm1)
xFlat_LSTM= Flatten()(xLstm2)

xConv = Conv1D(filters=30, kernel_size=8, strides=1, padding = 'valid', activation='relu')(CNN_x)
xPool = MaxPooling1D(pool_size=4, strides=1, padding='valid')(xConv)
xFlat_CNN = Flatten()(xPool)

Output_LSCNN = Dense(3, activation='linear')(concatenate([xFlat_LSTM,xFlat_CNN]))
model = Model([LSTM_x, CNN_x], Output_LSCNN)
model.compile(loss='mse', optimizer=Adam(lr=0.001))

model.fit([input_array_LSTM, input_array_CNN],output_array_2D, epochs=100)

pred_LSCNN= model.predict([test_array_LSCNN,test_array_LSCNN])
pred_LSCNN= pred_LSCNN[-20:,:]

ax1 = np.arange(1, len(scaled_df) + 1)
ax2 = np.arange(len(scaled_df), len(scaled_df) + len(pred_LSCNN))
plt.figure(figsize=(8, 3))
plt.plot(ax1, scaled_df, label='Time series', linewidth=1)
plt.plot(ax2, pred_LSCNN, label='Estimate')
plt.axvline(x=ax1[-1],  linestyle='dashed', linewidth=1)
plt.title('LSCNN')
plt.legend()
plt.show()


# 1-dimensional CNN
input_list= []
for i in range(nStep, len(by_date_df)):
    input_list.append(np.array(scaled_df[i-nStep:i]).reshape(-1,20,3))

output_list= []
for i in range(nStep+1, len(by_date_df)+1):
    output_list.append(np.array(scaled_df[i-nStep:i]).reshape(-1,20,3))

input_array_1D= np.array(input_list).reshape(-1,20,3)
output_array_1D= np.array(output_list).reshape(-1,3)[-141:]
test_array_1D= np.array(output_list).reshape(-1,20,3)

nStep = 20
nFeature = 3

xInput = Input(batch_shape = (None, nStep, nFeature))
xConv = Conv1D(filters=30, kernel_size=8, strides=1, padding = 'valid', activation='relu')(xInput)
xPool = MaxPooling1D(pool_size=4, strides=1, padding='valid')(xConv)
xFlat = Flatten()(xPool) # Latent Feature 생성(FC 이전 최종형태)
xOutput = Dense(3, activation='linear')(xFlat)
model = Model(xInput, xOutput)
model.compile(loss='mse', optimizer = optimizers.Adam(lr=0.001))

history1 = model.fit(input_array_1D, output_array_1D, epochs = 100, batch_size = 300)

pred_1d= model.predict(test_array_1D)
pred_1d= pred_1d[-20:,:]

ax1 = np.arange(1, len(scaled_df) + 1)
ax2 = np.arange(len(scaled_df), len(scaled_df) + len(pred_1d))
plt.figure(figsize=(8, 3))
plt.plot(ax1, scaled_df, label='Time series', linewidth=1)
plt.plot(ax2, pred_1d, label='Estimate')
plt.axvline(x=ax1[-1],  linestyle='dashed', linewidth=1)
plt.title('1D-CNN')
plt.legend()
plt.show()

# 2-dimensional CNN
input_list= []
for i in range(nStep, len(by_date_df)):
    input_list.append(np.array(scaled_df[i-nStep:i]).reshape(-1,20,3,1))

output_list= []
for i in range(nStep+1, len(by_date_df)+1):
    output_list.append(np.array(scaled_df[i-nStep:i]).reshape(-1,20,3,1))

input_array_2D= np.array(input_list).reshape(-1,20,3,1)
output_array_2D= np.array(output_list).reshape(-1,3)[-141:]
test_array_2D= np.array(output_list).reshape(-1,20,3,1)


nStep = 20
nFeature = 3
nChannel = 1

xInput = Input(batch_shape = (None, nStep, nFeature, nChannel))
xConv1 = Conv2D(filters=30, kernel_size=(2,2), strides=1, padding = 'same', activation='relu')(xInput) #동일유지
xPool1 = MaxPooling2D(pool_size=(2,2), strides=1, padding='valid')(xConv1)
xConv2 = Conv2D(filters=10, kernel_size=(2,2), strides=1, padding = 'same', activation='relu')(xPool1)
xPool2 = MaxPooling2D(pool_size=(2,2), strides=1, padding='valid')(xConv2)
xFlat = Flatten()(xPool2)
xOutput = Dense(3, activation='linear')(xFlat)
model = Model(xInput, xOutput)
model.compile(loss= 'mse', optimizer= Adam(lr=0.001))

history2= model.fit(input_array_2D,output_array_2D, epochs = 100)

pred_2d= model.predict(test_array)
pred_2d= pred_2d[-20:,:]

ax1 = np.arange(1, len(scaled_df) + 1)
ax2 = np.arange(len(scaled_df), len(scaled_df) + len(pred_2d))
plt.figure(figsize=(8, 3))
plt.plot(ax1, scaled_df, label='Time series', linewidth=1)
plt.plot(ax2, pred_2d, label='Estimate')
plt.axvline(x=ax1[-1],  linestyle='dashed', linewidth=1)
plt.title('2D-CNN')
plt.legend()
plt.show()
