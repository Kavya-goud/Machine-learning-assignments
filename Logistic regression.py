#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[21]:


data = pd.read_csv("C:\\Users\\Kavya\\Desktop\\DS Assignments\\ML\\bank-full.csv", delimiter=";",quotechar='"')


# In[22]:


data


# In[23]:


data.head()


# In[24]:


data.shape


# In[25]:


data.info()


# In[26]:


data.describe()


# In[28]:


data.hist()


# In[51]:


Y = data['y']


# In[52]:


X = data.iloc[:,0:16]


# In[53]:


X


# In[54]:


#label encoding
from sklearn.preprocessing import LabelEncoder
LE = LabelEncoder()


# In[55]:


categorical_columns = data.select_dtypes(include=['object']).columns.tolist()


# In[56]:


for col in categorical_columns:
    data[col] = LE.fit_transform(data[col])


# In[57]:


data


# In[58]:


# Standardization
from sklearn.preprocessing import StandardScaler
SS = StandardScaler()


# In[59]:


SS_X = SS.fit_transform(X)


# In[61]:


pd.DataFrame(SS_X)


# In[63]:


# model building
from sklearn.linear_model import LogisticRegression 
logreg = LogisticRegression()


# In[64]:


logreg.fit(SS_X,Y)


# In[65]:


Y_pred = logreg.predict(SS_X)


# In[70]:


Y_pred


# In[67]:


from sklearn.metrics import confusion_matrix,accuracy_score  
cm = confusion_matrix(Y,Y_pred)


# In[68]:


cm


# In[69]:


print("Accuracy score:", accuracy_score(Y,Y_pred).round(2))


# In[71]:


logreg.predict(X)


# In[ ]:




