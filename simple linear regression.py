#!/usr/bin/env python
# coding: utf-8

# In[20]:


import pandas as pd
import numpy as np
import seaborn as sns
import statsmodels.formula.api as smf


# In[12]:


df = pd.read_csv("C:\\data science\\delivery_time.csv")


# In[13]:


df


# In[14]:


# EDA
sns.distplot(df['Delivery Time'])


# In[15]:


sns.distplot(df['Sorting Time'])


# In[16]:


dataset=df.rename({'Delivery Time':'delivery_time', 'Sorting Time':'sorting_time'},axis=1)
dataset


# In[17]:


dataset.corr()


# In[18]:


sns.regplot(x=dataset['sorting_time'],y=dataset['delivery_time'])


# # MODEL TESTING

# In[21]:


model=smf.ols("delivery_time~sorting_time",data=dataset).fit()


# In[22]:


# MODEL BUILDING
# Finding Coefficient parameters
model.params


# In[23]:


# Finding tvalues and pvalues
model.tvalues , model.pvalues


# In[24]:


# Finding Rsquared Values
model.rsquared , model.rsquared_adj


# # Model Predictions

# In[25]:


delivery_time = (6.582734) + (1.649020)*(5)


# In[26]:


delivery_time


# # Question - 2

# In[27]:


import pandas as pd
import numpy as np
import seaborn as sns
import statsmodels.formula.api as smf


# In[28]:


df = pd.read_csv("C:\\data science\\Salary_Data.csv")


# In[29]:


df


# In[30]:


# EDA
df.info()


# In[31]:


sns.distplot(df['YearsExperience'])


# In[32]:


sns.distplot(df['Salary'])


# In[33]:


df.corr()


# In[34]:


sns.regplot(x=df['YearsExperience'],y=df['Salary'])


# # Model Building

# In[35]:


model=smf.ols("Salary~YearsExperience",data=df).fit()


# In[36]:


# Model Testing
# Finding Co-efficient Parameters
model.params


# In[37]:


# Finding Pvalues and tvalues
model.tvalues, model.pvalues


# In[38]:


# Finding Pvalues and tvalues
model.tvalues, model.pvalues


# In[39]:


# Finding Rsquared values
model.rsquared , model.rsquared_adj


# In[41]:


# MODEL PREDICTIONS
# Manual prediction for say 3 Years Experience
Salary = (25792.200199) + (9449.962321)*(3)


# In[42]:


Salary


# In[ ]:




