#!/usr/bin/env python
# coding: utf-8

# In[2]:


# !pip install yfinance


# In[35]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


# In[36]:


import yfinance as yt


# In[37]:


start_date = '2010-01-01'
end_date = '2023-07-01'

df = yt.download("TSLA" , start = start_date,end = end_date).reset_index()


# In[38]:


df


# In[39]:


df.head()


# # Spiting the data Train and Validation

# In[40]:


data = df['Open'].tolist()
lenght_data = len(data)
split_ratio = 0.7
len_train = round(lenght_data*split_ratio)
len_train,lenght_data


# In[41]:


train_data = data[:len_train]
test_data = data[len_train:]


# In[49]:


from sklearn.preprocessing import MinMaxScaler #MinMax Scaler put the value 0 and 1
st = MinMaxScaler(feature_range=(0,1))
train_data = np.array(train_data).reshape(-1,1)
test_data = np.array(test_data).reshape(-1,1)


# In[50]:


train_data.shape , test_data.shape


# In[97]:


train_scaled = st.fit_transform(train_data).flatten()
test_scaled = st.fit_transform(test_data).flatten()


# In[98]:


train_scaled


# In[99]:


window_size = 50
x, y = [], []
data = train_scaled

def window_datset(data,window_size):
    for i in range(len(data)-window_size-1):
        x.append(data[i:(i+window_size)])
        y.append(data[i+window_size])
    return np.array(x), np.array(y)


# In[100]:


x_train ,y_train = window_datset(train_scaled,window_size)
x_test,y_test = window_datset(test_scaled,window_size)


# In[101]:


x_train


# In[102]:


y_train


# # Creating RNN Model

# In[106]:


import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense,LSTM,Dropout


# In[107]:


model = Sequential()
model.add(LSTM(units= window_size,activation='tanh',return_sequences=True,
              input_shape =(x_train.shape[1],1)))
model.add(Dropout(0.2))

model.add(LSTM(units= window_size,activation='tanh',return_sequences=True))
model.add(Dropout(0.2))

model.add(LSTM(units= window_size,activation='tanh'))

model.add(Dense(1))


# In[108]:


model.summary()


# In[109]:


model.compile(loss =tf.keras.losses.Huber(), optimizer = 'sgd',metrics = ['mae'])


# In[110]:


history = model.fit(x_train,y_train , epochs=50,batch_size = 32)


# # Evaluate The Model

# In[111]:


y_pred = model.predict(x_test)
y_pred = st.inverse_transform(y_pred)  #inverse_transform convert iur data MinMaxscaling to original value


# In[112]:


y_test =st.inverse_transform(y_test.reshape(-1,1))


# In[113]:


y_test


# In[117]:


plt.plot(y_pred,color = 'b',label ='y_pred')
plt.plot(y_test,color ='g',label = 'y_test')
plt.xlabel("Days")
plt.ylabel("Open Price")
plt.title("Prediction with X_test vs Y_test")
plt.legend()
plt.show()


# In[118]:


import pickle


# In[120]:


model.save('tesla.h5')
pickle.dump(st,open('scaler.pkl','wb'))


# In[121]:


from keras.models import load_model


# In[122]:


model1 = load_model('tesla.h5')
scaler1 = pickle.load(open('scaler.pkl','rb'))


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




