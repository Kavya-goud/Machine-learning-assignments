#!/usr/bin/env python
# coding: utf-8

# In[43]:


import pandas as pd
import numpy as np
from sklearn.preprocessing import scale


# In[44]:


data = pd.read_csv("C:\\Users\\Kavya\\Desktop\\DS Assignments\\ML\\PCA\\wine.csv")


# In[45]:


data


# In[46]:


data.shape


# In[47]:


data.head()


# In[24]:


# split as X and Y variable
Y = data["Type"]
X = data.iloc[:,1:]


# In[25]:


Y


# In[26]:


X


# In[27]:


# standardization
from sklearn.preprocessing import StandardScaler
SS = StandardScaler()
SS_X = SS.fit_transform(X)
pd.DataFrame(SS_X)


# In[28]:


# using KNN
# importing pca
from sklearn.decomposition import PCA
pc = PCA()


# In[29]:


pc_df  = pc.fit_transform(SS_X)
pc_df = pd.DataFrame(pc_df)


# In[30]:


pc_df


# In[31]:


pd.DataFrame((pc.explained_variance_ratio_)*100)


# In[34]:


import matplotlib.pyplot as plt
plt.scatter(range(0,13),pd.DataFrame((pc.explained_variance_ratio_)*100))
plt.show()


# In[35]:


X_new = pc_df.iloc[:,0:7]


# In[36]:


X_new


# In[81]:


from sklearn.neighbors import KNeighborsClassifier
KNN = KNeighborsClassifier(n_neighbors=3)


# In[82]:


k1_train = []
k1_test = []


# In[83]:


from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
training_accuracy = []
test_accuracy = []

for i in range(1,100):
    X_train,X_test,Y_train,Y_test = train_test_split(X_new,Y,test_size=0.30,random_state=i)
    KNN.fit(X_train,Y_train)
    Y_pred_train = KNN.predict(X_train)
    Y_pred_test  = KNN.predict(X_test)
    training_accuracy.append(accuracy_score(Y_train ,Y_pred_train))
    test_accuracy.append(accuracy_score(Y_test ,Y_pred_test))


# In[84]:


import numpy as np
print("Average training accuracy:",np.mean(training_accuracy).round(3))
print("Average test accuracy:",np.mean(test_accuracy).round(3))


# # Heirarchial clustering

# In[49]:


list(data)


# In[50]:


import scipy.cluster.hierarchy as sch
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import normalize


# In[52]:


# As we already have normalized data, create Dendrograms
plt.figure(figsize=(10,8))
dendrogram=sch.dendrogram(sch.linkage(SS_X,'complete'))


# In[77]:


# forming a group using cluster
from sklearn.cluster import AgglomerativeClustering
cluster = AgglomerativeClustering(n_clusters=6, metric ='euclidean', linkage = 'complete')


# In[78]:


Y = cluster.fit_predict(X)


# In[79]:


Y = pd.DataFrame(Y)


# In[80]:


Y.value_counts()


# In[ ]:




