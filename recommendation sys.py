#!/usr/bin/env python
# coding: utf-8

# In[46]:


import pandas as pd
import numpy as np


# In[2]:


df = pd.read_csv("C:\\Users\\Kavya\\Desktop\\DS Assignments\\ML\\Recommendation system\\book.csv", encoding='latin1')


# In[3]:


df


# In[5]:


df.head()


# In[4]:


df.tail()


# In[6]:


df.shape


# In[7]:


df.info()


# In[8]:


df.isnull().sum()


# In[10]:


df.drop(df.columns[0],axis=1,inplace=True)


# In[11]:


df


# In[13]:


# Renaming the columns
df.columns = ["User_ID","Book_Title","Book_Rating"]


# In[14]:


df


# In[16]:


df = df.sort_values(by=['User_ID'])


# In[17]:


df


# In[18]:


df.nunique()


# In[21]:


df.loc[df["Book_Rating"] == 'small', 'Book_Rating'] = 0
df.loc[df["Book_Rating"] == 'large', 'Book_Rating'] = 1


# In[22]:


df.Book_Rating.value_counts()


# In[26]:


import matplotlib.pyplot as plt
import seaborn as sns
plt.figure(figsize=(20,6))
sns.histplot(df.Book_Rating)


# In[27]:


book_df = df.pivot_table(index='User_ID',
                   columns='Book_Title',
                   values='Book_Rating').reset_index(drop=True)


# In[28]:


book_df.fillna(0,inplace=True)


# In[29]:


book_df


# In[33]:


# Average rating of books
avg = df['Book_Rating'].mean()


# In[34]:


avg


# In[35]:


# Calculate the minimum number of votes required to be in the chart, 
minimum = df['Book_Rating'].quantile(0.90)
minimum


# In[38]:


# Filter out all qualified Books into a new DataFrame
q_Books = df.copy().loc[df['Book_Rating'] >= minimum]
q_Books.shape


# # Calculating Cosine Similarity between Users

# In[39]:


from sklearn.metrics import pairwise_distances
from scipy.spatial.distance import cosine,correlation


# In[40]:


user_sim=1-pairwise_distances(book_df.values,metric='cosine')


# In[41]:


user_sim


# In[42]:


user_sim_df=pd.DataFrame(user_sim)


# In[43]:


user_sim_df


# In[51]:


#Set the index and column names to user ids 
user_sim_df.index = df.User_ID.unique()
user_sim_df.columns = df.User_ID.unique()


# In[52]:


user_sim_df


# In[53]:


np.fill_diagonal(user_sim,0)
user_sim_df


# In[60]:


#Most Similar Users
print(user_sim_df.idxmax(axis=1))
print(user_sim_df.max(axis=1).sort_values(ascending=False).head(50))


# In[74]:


reader = df[(df['User_ID']==1348) | (df['User_ID']==2576)]
reader


# In[76]:


reader1=df[(df['User_ID']==1348)] 


# In[77]:


reader1


# In[78]:


reader2=df[(df['User_ID']==2576)] 


# In[79]:


reader2


# In[ ]:




