import pandas as pd
import datetime
from datetime import datetime
from tensorflow.keras.layers import Dense, Input, LSTM, TimeDistributed,Bidirectional
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import PowerTransformer
import numpy as np
import matplotlib.pyplot as plt

data= pd.read_csv('dataset/201901-202003.csv')
data.dtypes

# astype dataset
data.REG_YYMM= data.REG_YYMM.astype('category')
data.CARD_SIDO_NM= data.CARD_SIDO_NM.astype('category')
data.CARD_CCG_NM= data.CARD_CCG_NM.astype('category')
data.STD_CLSS_NM= data.STD_CLSS_NM.astype('category')
data.HOM_SIDO_NM= data.HOM_SIDO_NM.astype('category')
data.HOM_CCG_NM= data.HOM_CCG_NM.astype('category')
data.AGE= data.AGE.astype('category')
data.SEX_CTGO_CD= data.SEX_CTGO_CD.astype('category')
data.FLC= data.FLC.astype('category')

# Transformation
pt= PowerTransformer(method='box-cox', standardize=False)
pt.fit(data.iloc[:,9:12])
pt_int_data= pt.transform(data.iloc[:,9:12])
pt.lambda_

# Group by
category_data= pd.DataFrame(data.iloc[:,:9])
pt_int_data= pd.DataFrame(pt_int_data, columns= data.columns[9:12])
pt_data= pd.concat([category_data, pt_int_data], axis= 1)
pt_data= pt_data.sort_values(by= 'REG_YYMM')

groupby_pt= pt_data.groupby(list(data.columns), observed= True)
sum_groupby_pt= groupby_pt.sum()

pt_data.REG_YYMM.value_counts()

# Shaping
nStep= 1000
input_list= []
for i in range(nStep, len(pt_data)):
    input_list.append(np.array(pt_data[i-nStep:i]).reshape(-1,1000,3))

output_list= []
for i in range(nStep+nStep, len(pt_data)+nStep):
    output_list.append(np.array(pt_data[i-nStep:i]).reshape(-1,1000,3))
    
input_array= np.array(input_list).reshape(-1,1000,3)
output_array= np.array(output_list).reshape(-1,1000,3)

# Modeling
nInput = 3
nOutput = 3
nStep = 1000
nHidden = 100

xInput = Input(batch_shape=(None, nStep, 3))
xLstm1 = LSTM(nHidden, return_sequences=True)(xInput)
xLstm2 = Bidirectional(LSTM(nHidden), merge_mode='concat')(xLstm1)
xOutput = Dense(nOutput)(xLstm2)
model = Model(xInput, xOutput)
model.compile(loss='mse', optimizer=Adam(lr=0.01))

model.fit(input_array, output_array)

# Predicting data
pred= model.predict(output_array)











