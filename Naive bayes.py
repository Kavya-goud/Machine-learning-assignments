#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np 
import pandas as pd


# In[2]:


train = pd.read_csv('C:\\Users\\Kavya\\Desktop\\DS Assignments\\ML\\Naive bayes\\SalaryData_Train.csv')


# In[3]:


train


# In[4]:


test = pd.read_csv("C:\\Users\\Kavya\\Desktop\\DS Assignments\\ML\\Naive bayes\\SalaryData_Test.csv")


# In[5]:


test


# In[6]:


train.info()


# In[8]:


train.describe()


# In[7]:


test.info()


# In[9]:


test.describe()


# In[10]:


train[train.duplicated()].shape


# In[11]:


train[train.duplicated()]


# In[12]:


Train =train.drop_duplicates()


# In[13]:


Train


# In[14]:


Train.isnull().sum().sum()
## there is no nan values in the Train Data set


# In[15]:


test[test.duplicated()].shape


# In[16]:


test[test.duplicated()]


# In[17]:


Test=test.drop_duplicates()


# In[18]:


Test


# In[19]:


Test.isnull().sum().sum()
## there is no nan values in the Train Data set


# In[20]:


Train['Salary'].value_counts()


# In[21]:


Test['Salary'].value_counts()


# In[22]:


pd.crosstab(Train['occupation'],Train['Salary'])


# In[23]:


pd.crosstab(Train['workclass'],Train['Salary'])


# In[24]:


pd.crosstab(Train['workclass'],Train['occupation'])


# In[25]:


import matplotlib.pyplot as plt
import seaborn as sns
sns.countplot(x='Salary',data= Train)
plt.xlabel('Salary')
plt.ylabel('count')
plt.show()
Train['Salary'].value_counts()


# In[26]:


sns.countplot(x='Salary',data= Test)
plt.xlabel('Salary')
plt.ylabel('count')
plt.show()
Test['Salary'].value_counts()


# In[33]:


pd.crosstab(Train['Salary'],Train['education']).mean().plot(kind='bar')


# In[35]:


pd.crosstab(Train['Salary'],Train['occupation']).mean().plot(kind='bar')


# In[36]:


pd.crosstab(Train['Salary'],Train['workclass']).mean().plot(kind='bar')


# In[37]:


pd.crosstab(Train['Salary'],Train['sex']).mean().plot(kind='bar')


# In[38]:


pd.crosstab(Train['Salary'],Train['relationship']).mean().plot(kind='bar')


# In[39]:


string_columns = ["workclass","education","maritalstatus","occupation","relationship","race","sex","native"]


# In[59]:


# Label Encoding
##Preprocessing the data. As, there are categorical variables
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
number = LabelEncoder()
for i in string_columns:
        Train[i]= number.fit_transform(Train[i])
        Test[i]=number.fit_transform(Test[i])


# In[60]:


Train


# In[61]:


Test


# In[62]:


##Capturing the column names which can help in futher process
colnames = Train.columns
colnames


# In[63]:


len(colnames)


# In[64]:


# Data Partition
from sklearn.model_selection import train_test_split,cross_val_score
x_train = Train[colnames[0:13]].values
y_train = Train[colnames[13]].values
x_test = Test[colnames[0:13]].values
y_test = Test[colnames[13]].values


# In[65]:


##Normalmization
def norm_func(i):
    x = (i-i.min())/(i.max()-i.min())
    return (x)


# In[66]:


x_train


# In[67]:


x_test


# In[68]:


y_train


# In[69]:


y_test


# In[70]:


x_train = norm_func(x_train)
x_test =  norm_func(x_test)


# # Applying naive bayes for classification

# In[71]:


# Preparing a naive bayes model on training data set 
from sklearn.naive_bayes import MultinomialNB as MB
M_model=MB()
train_pred_multi=M_model.fit(x_train,y_train).predict(x_train)
test_pred_multi=M_model.fit(x_train,y_train).predict(x_test)


# In[72]:


train_acc_multi=np.mean(train_pred_multi==y_train)
train_acc_multi ## train accuracy 74.42


# In[73]:


test_acc_multi=np.mean(test_pred_multi==y_test)
test_acc_multi ## test acuracy 75.15


# In[74]:


# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(y_test, test_pred_multi)


# In[75]:


confusion_matrix


# In[76]:


#calculating the accuracy of this model w.r.t. this dataset
from sklearn.metrics import accuracy_score
print(accuracy_score(y_test,test_pred_multi))


# In[ ]:




