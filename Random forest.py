#!/usr/bin/env python
# coding: utf-8

# # QUESTION - 1(company.csv file) 

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


df = pd.read_csv("C:\\Users\\Kavya\\Desktop\\DS Assignments\\ML\\Random forest\\Company_Data.csv")


# In[3]:


df


# In[4]:


df.info()


# In[5]:


list(df)


# In[7]:


#EDA
import matplotlib.pyplot as plt
import seaborn as sns
sns.pairplot(data = df, hue = 'Sales')


# In[8]:


# LABEL ENCODING
# Convert categorical variables to numerical using dummy variables
df = pd.get_dummies(df, columns=['Urban', 'US'], drop_first=True)


# In[9]:


# Map 'ShelveLoc' values to numerical values
df['ShelveLoc'] = df['ShelveLoc'].map({'Good': 1, 'Medium': 2, 'Bad': 3})


# In[13]:


#features (X) and target (y)
X = df.drop('Sales', axis=1)


# In[14]:


X


# In[25]:


y = df['Sales']


# In[26]:


y


# In[27]:


# Split the data into training and testing sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)


# In[28]:


#Random Forest Regressor
from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor(n_estimators=1000, random_state=42)


# In[29]:


rf.fit(X_train,y_train)


# In[30]:


predictions = rf.predict(X_test)


# In[31]:


# Mean absolute error and accuracy
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
mae = mean_absolute_error(y_test, predictions)


# In[32]:


mape = mean_absolute_percentage_error(y_test, predictions)


# In[33]:


accuracy = 100 - mape * 100


# In[34]:


print(f"Mean Absolute Error: {round(mae, 2)}")
print(f"Accuracy: {round(accuracy, 2)}%")


# # QUESTION - 2(fraud check.csv)

# In[20]:


import pandas as pd 
import numpy as np


# In[21]:


df = pd.read_csv("C:\\Users\\Kavya\\Desktop\\DS Assignments\\ML\\Random forest\\Fraud_Check.csv")


# In[22]:


df


# In[23]:


df.info()


# In[24]:


df = df.rename({'Undergrad':'under_grad', 'Marital.Status':'marital_status', 'Taxable.Income':'taxable_income',
                    'City.Population':'city_population', 'Work.Experience':'work_experience', 'Urban':'urban'}, axis = 1)


# In[25]:


df


# In[26]:


df['taxable_income'] = df.taxable_income.map(lambda x: 1 if x <= 30000 else 0)


# In[27]:


df


# In[28]:


# Label encoding
from sklearn.preprocessing import LabelEncoder
LE = LabelEncoder()


# In[29]:


df['under_grad'] = LE.fit_transform(df['under_grad'])
df['marital_status'] = LE.fit_transform(df['marital_status'])
df['urban'] = LE.fit_transform(df['urban'])


# In[30]:


df


# In[31]:


Y = df['taxable_income']


# In[32]:


X = df[["under_grad","marital_status","city_population","work_experience","urban"]]


# In[33]:


# Random forest
from sklearn.ensemble import RandomForestClassifier
RFC = RandomForestClassifier(n_estimators=100,max_samples=0.6,max_features=0.7)


# In[34]:


RFC.fit(X,Y)


# In[36]:


Y_pred = RFC.predict(X)


# In[37]:


from sklearn.metrics import accuracy_score
ac = accuracy_score(Y,Y_pred)
print(ac)


# In[38]:


from sklearn.model_selection import train_test_split

training_accuracy = []
test_accuracy = []

for i in range(1,100):
    X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.30,random_state=i)
    RFC.fit(X_train,Y_train)
    Y_pred_train = RFC.predict(X_train)
    Y_pred_test  = RFC.predict(X_test)
    training_accuracy.append(accuracy_score(Y_train ,Y_pred_train))
    test_accuracy.append(accuracy_score(Y_test ,Y_pred_test))


# In[39]:


print("Average training accuracy:",np.mean(training_accuracy).round(3))
print("Average test accuracy:",np.mean(test_accuracy).round(3))


# In[ ]:




