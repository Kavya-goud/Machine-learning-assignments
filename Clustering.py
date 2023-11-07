#!/usr/bin/env python
# coding: utf-8

# # Crime-data(Question - 1)

# In[44]:


import pandas as pd


# In[45]:


df = pd.read_csv("C:\\Users\\Kavya\\Desktop\\DS Assignments\\\ML\\crime_data.csv")


# In[46]:


df


# In[47]:


df.shape


# In[48]:


df.head()


# In[49]:


list(df)


# In[50]:


X = df.iloc[:,1:]


# In[51]:


import scipy.cluster.hierarchy as shc


# In[52]:


# construction of dendogram
import matplotlib.pyplot as plt
plt.figure(figsize=(10,7))
plt.title("Customer Dendograms")
dend = shc.dendrogram(shc.linkage(X, method = 'complete'))


# In[53]:


# forming a group using cluster
from sklearn.cluster import AgglomerativeClustering
cluster = AgglomerativeClustering(n_clusters=3, metric ='euclidean', linkage = 'complete')


# In[54]:


Y = cluster.fit_predict(X)


# In[55]:


Y = pd.DataFrame(Y)


# In[56]:


Y.value_counts()


# # K-means

# In[356]:


from sklearn.cluster import KMeans
KMeans = KMeans(n_clusters= 3,n_init= 30)


# In[357]:


KMeans.fit(X)


# In[358]:


Y = KMeans.predict(X)


# In[359]:


Y = pd.DataFrame(Y)


# In[360]:


Y[0].value_counts()


# In[361]:


KMeans.inertia_


# In[362]:


inertia = []


# In[363]:


from sklearn.cluster import KMeans
for i in range(1,11):
    km = KMeans(n_clusters=i , random_state=0)
    km.fit(X)
    inertia.append(km.inertia_)


# In[364]:


plt.scatter(range(1,11), inertia)
plt.plot(range(1,11), inertia,color = 'red')
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('inertia')
plt.show()


# # DBSCAN

# In[57]:


X = df.iloc[:,1:]


# In[58]:


X


# In[59]:


# Standardization
from sklearn.preprocessing import StandardScaler
SS = StandardScaler()
SS_X = SS.fit_transform(X)


# In[60]:


from sklearn.cluster import DBSCAN
DBSCAN()


# In[61]:


dbscan = DBSCAN(eps=0.75, min_samples=3)
dbscan.fit(SS_X)


# In[62]:


#Noisy samples are given the label as -1
dbscan.labels_
c1 = pd.DataFrame(dbscan.labels_,columns=['cluster'])
print(c1['cluster'].value_counts())


# In[63]:


clustered = pd.concat([df,c1],axis=1)


# In[64]:


noisedata = clustered[clustered['cluster']==-1]


# In[65]:


noisedata


# In[66]:


finaldata = clustered[clustered['cluster']==0]


# In[67]:


finaldata


# In[68]:


from sklearn.cluster import KMeans
KMeans = KMeans(n_clusters = 5,n_init=30)
KMeans.fit(finaldata.iloc[:,1:])


# In[69]:


Y = KMeans.predict(finaldata.iloc[:,1:])


# In[70]:


Y = pd.DataFrame(Y)


# In[71]:


Y[0].value_counts()


# In[ ]:





# # Eastwest Airlines(Question - 2)

# In[72]:


import pandas as pd


# In[73]:


df = pd.read_csv("C:\\Users\\Kavya\\Desktop\\DS Assignments\\ML\\EastWestAirlines.csv")


# In[74]:


df


# In[75]:


df.shape


# In[76]:


df.head()


# In[77]:


list(df)


# In[78]:


X = df.iloc[:,1:]
X


# In[8]:


import scipy.cluster.hierarchy as sch


# In[9]:


from sklearn.preprocessing import normalize
df_norm = pd.DataFrame(normalize(df),columns=df.columns)
df_norm


# In[249]:


plt.figure(figsize=(10,7))
dendrograms=sch.dendrogram(sch.linkage(df_norm, 'complete'))


# In[10]:


# forming a group using cluster
from sklearn.cluster import AgglomerativeClustering
cluster = AgglomerativeClustering(n_clusters=5, metric ='euclidean', linkage = 'ward')


# In[11]:


Y = cluster.fit_predict(X)


# In[12]:


Y = pd.DataFrame(Y)


# In[13]:


Y.value_counts()


# # K means

# In[16]:


from sklearn.cluster import KMeans


# In[17]:


KMeans = KMeans(n_clusters = 3,n_init=10)


# In[18]:


KMeans.fit(X)


# In[19]:


Y = KMeans.predict(X)


# In[20]:


Y = pd.DataFrame(Y)


# In[21]:


Y[0].value_counts()


# In[22]:


KMeans.inertia_


# In[23]:


inertia = []


# In[24]:


from sklearn.cluster import KMeans
for i in range(1,11):
    km = KMeans(n_clusters=i , random_state=0)
    km.fit(X)
    inertia.append(km.inertia_)


# In[26]:


import matplotlib.pyplot as plt
plt.scatter(range(1,11), inertia)
plt.plot(range(1,11), inertia,color = 'red')
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('inertia')
plt.show()


# # DBSCAN

# In[79]:


X = df.iloc[:,1:]


# In[80]:


X


# In[81]:


from sklearn.preprocessing import StandardScaler
SS = StandardScaler()
SS_X = SS.fit_transform(X)


# In[82]:


from sklearn.cluster import DBSCAN
DBSCAN()


# In[83]:


dbscan = DBSCAN(eps=0.75, min_samples=3)
dbscan.fit(SS_X)


# In[84]:


#Noisy samples are given the label as -1
dbscan.labels_
c1 = pd.DataFrame(dbscan.labels_,columns=['cluster'])
print(c1['cluster'].value_counts())


# In[85]:


clustered = pd.concat([df,c1],axis=1)


# In[86]:


noisedata = clustered[clustered['cluster']==-1]


# In[87]:


noisedata


# In[88]:


finaldata = clustered[clustered['cluster']==0]


# In[89]:


finaldata


# In[90]:


from sklearn.cluster import KMeans
KMeans = KMeans(n_clusters = 5,n_init=30)
KMeans.fit(finaldata.iloc[:,1:])


# In[91]:


Y = KMeans.predict(finaldata.iloc[:,1:])


# In[92]:


Y = pd.DataFrame(Y)


# In[93]:


Y[0].value_counts()


# In[ ]:




