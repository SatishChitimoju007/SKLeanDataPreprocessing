#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[5]:


df = pd.read_csv("C:/OhmNaamahShivaya/sklearning/auto-mpg.csv")


# In[6]:


df.head()


# In[7]:


df.tail()


# In[8]:


df.shape


# In[10]:


df.dtypes


# In[11]:


X = df.iloc[:,[2,4]]
y = df.iloc[:,5]


# In[12]:


X.head()


# In[16]:


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# In[17]:


X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 40)


# In[18]:


X_train.head()


# In[19]:


scaler = StandardScaler().fit(X_train)


# In[20]:


print(scaler)


# In[21]:


scaler.mean_


# In[22]:


scaler.scale_


# In[23]:


scaler.transform(X_train)


# In[26]:


X_train_scaled = scaler.transform(X_train)


# In[27]:


print(X_train_scaled)


# In[28]:


print(X_train_scaled.mean(axis=0))


# In[33]:


print(X_train_scaled.std(axis=0))


# In[32]:


scaler = StandardScaler().fit(X_test)


# In[35]:


X_test.head()


# In[31]:


scaler.mean_


# In[36]:


scaler.scale_


# In[37]:


scaler.transform(X_test)


# In[38]:


X_test_scaled = scaler.transform(X_test)


# In[39]:


print(X_test_scaled.mean(axis=0))


# In[40]:


print(X_test_scaled.std(axis=0))


# In[ ]:




