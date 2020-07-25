import pandas as pd
import datetime
from datetime import datetime
from tensorflow.keras.layers import Dense, Input, LSTM, TimeDistributed,Bidirectional
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

# 일별로 가공한 dataframe 정규화하기
scaled_df= scale(by_date_df,axis=0) # feature별로 정규화

# LSTM형태로 data 가공하기
nInput = 3
nOutput = 3
nStep = 20
nHidden = 50

input_list= []
for i in range(nStep, len(by_date_df)):
    input_list.append(np.array(scaled_df[i-nStep:i]).reshape(-1,20,3))

output_list= []
for i in range(nStep+1, len(by_date_df)+1):
    output_list.append(np.array(scaled_df[i-nStep:i]).reshape(-1,20,3))
    
input_array= np.array(input_list).reshape(-1,20,3)
output_array= np.array(output_list).reshape(-1,20,3)

# LSTM modeling
xInput = Input(batch_shape=(None, nStep, 3))
xLstm1 = LSTM(nHidden, return_sequences=True)(xInput)
xLstm2 = Bidirectional(LSTM(nHidden), merge_mode='concat')(xLstm1)
xOutput = Dense(nOutput)(xLstm2)
model = Model(xInput, xOutput)
model.compile(loss='mse', optimizer=Adam(lr=0.01))
history = model.fit(input_array, scaled_df[-141:,:], epochs = 100, shuffle=True)

# predicting
estimate= []
for i in range(20):
    estimate.append(model.predict(np.array(by_date_df.iloc[i+len(scaled_df)-40,i+len(scaled_df)-20:]).reshape(-1,20,3)))
pred1_pd= pd.DataFrame(pred1.reshape(-1,3))

# plotting
ax1 = np.arange(1, len(scaled_df) + 1)
ax2 = np.arange(len(scaled_df), len(scaled_df) + len(pred1_pd))
plt.figure(figsize=(8, 3))
plt.plot(ax1, scaled_df, label='Time series', linewidth=1)
plt.plot(ax2, pred1_pd, label='Estimate')
plt.axvline(x=ax1[-1],  linestyle='dashed', linewidth=1)
plt.legend()
plt.show()
