#!/usr/bin/env python
# coding: utf-8

# # Question - 1

# In[41]:


import numpy as np 
import pandas as pd
import seaborn as sns
import scipy.stats as stat
import statsmodels.api as smf
import statsmodels.formula.api as sm


# In[42]:


df = pd.read_csv("C:\\data science\\50_Startups.csv")


# In[43]:


df


# In[44]:


df.shape


# In[45]:


df.describe()


# In[46]:


df.info()


# In[47]:


df.isnull().sum()


# In[48]:


#EDA
import matplotlib.pyplot as plt
plt.scatter(df['Profit'],df['R&D Spend'],color= 'red')
plt.show()


# In[49]:


plt.scatter(df['Profit'],df['Administration'],color= 'red')
plt.show()


# In[50]:


plt.scatter(df['Profit'],df['Administration'],color= 'red')
plt.show()


# In[51]:


plt.scatter(df['Profit'],df['Marketing Spend'],color= 'red')
plt.show()


# In[52]:


plt.scatter(df['Profit'],df['State'],color= 'red')
plt.show()


# In[53]:


df.corr()


# In[54]:


Y = df['Profit']


# In[55]:


Y


# In[56]:


X = df.iloc[:,0:4]


# In[57]:


X


# In[58]:


# Data Transformation
# Label Encoding
from sklearn.preprocessing import LabelEncoder  
LE = LabelEncoder()


# In[59]:


X['State'] = LE.fit_transform(df['State'])


# In[60]:


X


# In[61]:


# Standardization
from sklearn.preprocessing import StandardScaler
SS = StandardScaler()


# In[62]:


X["R&D Spend"] = SS.fit_transform(df[['R&D Spend']])
X["Administration"] = SS.fit_transform(df[['Administration']])
X["Marketing Spend"] = SS.fit_transform(df[['Marketing Spend']])


# In[63]:


X


# In[64]:


SS_X = SS.fit_transform(X)


# In[66]:


pd.DataFrame(SS_X)


# In[67]:


rmse_test=[]
r2_scores=[]

for i in range(1,11):
    from sklearn.model_selection import train_test_split
    X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.30,random_state=i)
    from sklearn.linear_model import LinearRegression
    LE=LinearRegression()
    LE.fit(X_train,Y_train)
    Y_pred_test=LE.predict(X_test)
    from sklearn.metrics import mean_squared_error,r2_score
    rmse_test.append(np.sqrt(mean_squared_error(Y_test,Y_pred_test)))
    r2_scores.append(r2_score(Y_test,Y_pred_test))


# In[68]:


print("RMSE of test :",np.mean(rmse_test).round(3))
print("r2_square :",np.mean(r2_scores).round(3))
r2_table=pd.DataFrame({"model":np.arange(1,11 ),"R2_score":r2_scores})
print(r2_table)


# # QUESTION - 2

# In[1]:


# Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.formula.api as smf
import statsmodels.api as sm


# In[2]:


df = pd.read_csv("C:\\Users\\Kavya\\Desktop\\DS Assignments\\ML\\ToyotaCorolla.csv", encoding='latin1')


# In[3]:


df


# In[4]:


# EDA
df.info()


# In[5]:


df_1 = pd.concat([df.iloc[:,2:4],df.iloc[:,6:7],df.iloc[:,8:9],df.iloc[:,12:14],df.iloc[:,15:18]],axis=1)


# In[6]:


df_1


# In[7]:


df_2=df_1.rename({'Age_08_04':'Age','cc':'CC','Quarterly_Tax':'QT'},axis=1)


# In[8]:


df_2


# In[9]:


df_2[df_2.duplicated()]


# In[10]:


df_3=df_2.drop_duplicates().reset_index(drop=True)


# In[11]:


df_3


# In[12]:


df_3.describe()


# In[13]:


df_3.corr()


# In[14]:


sns.set_style(style='darkgrid')
sns.pairplot(df_3)


# In[34]:


Y = df_3['Price']


# In[35]:


Y


# In[36]:


X = df_3.iloc[:,1:]


# In[37]:


X


# In[91]:


Y = df_3['Price']


# In[85]:


#X = df_3[['Age']]


# In[92]:


from sklearn.linear_model import LinearRegression
LR = LinearRegression()


# In[93]:


LR.fit(X,Y)


# In[94]:


Y_pred = LR.predict(X)


# In[95]:


from sklearn.metrics import mean_squared_error                      # step:4 fit the model
mse = mean_squared_error(Y,Y_pred)
print("mean squared error:", mse.round(3))
print("Root mean squared error:", np.sqrt(mse).round(3))


# In[90]:


X = df_3[['Age','KM','HP','CC','Doors','Gears','QT','Weight']]


# Using stats model

# In[97]:


model = smf.ols('Price ~ Age+KM+HP+CC+Doors+Gears+QT+Weight', data = df_3).fit()


# In[98]:


model.summary()


# In[99]:


# Predicted values
model.fittedvalues


# In[100]:


model.resid


# In[101]:


mse1 = np.mean(model.resid ** 2)
print('mean square error', mse1)
print("Root mean squared error:", np.sqrt(mse1).round(3))

