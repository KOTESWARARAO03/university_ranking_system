#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans


# In[2]:


Univ = pd.read_csv("Universities.csv")
Univ.head()


# ### Standardizing the data

# In[3]:


def std_data(i):
    z=(i-i.mean())/i.std()
    return z


# In[4]:


std_Univ_data=std_data(Univ.iloc[:,1:])
std_Univ_data


# ### How to find optimum number of  cluster
# ### The K-means algorithm aims to choose centroids that minimise the inertia, or within-cluster sum-of-squares criterion:

# In[5]:


kmeans_cluster=KMeans(n_clusters=2)
kmeans_cluster.fit(std_Univ_data)


# In[6]:


kmeans_cluster.labels_


# In[7]:


#total sum of squares of within cluster distances (for all k clusters)-inertia
#this is for k=2
kmeans_cluster.inertia_


# ### Checking for the optimal k value

# ### Elbow plot/Scree plot-->plot x and y to get the optimal k value
# #### y- inertia values corresponding to k values
# #### x-different k values

# In[8]:


inertia=[]
k= (list(range(2,8)))
for i in k:
    kmeans_cluster=KMeans(n_clusters=i)
    kmeans_cluster.fit(std_Univ_data)
    inertia.append(kmeans_cluster.inertia_)
    print(i,inertia)


# In[9]:


k


# In[10]:


#elbow plot
plt.plot(k,inertia,marker=".")
plt.title("Scree plot")
plt.xlabel("k")
plt.ylabel("Inertia")


# ## We can see that k=3 or k=5 are the optimal values of k.

# In[11]:


#taking k=3
kmeans_cluster_3=KMeans(n_clusters=3)
kmeans_cluster_3.fit(std_Univ_data)


# In[12]:


labels_3=kmeans_cluster_3.labels_
labels_3


# In[13]:


kmeans_cluster_3.inertia_


# In[14]:


#taking k=5
kmeans_cluster_5=KMeans(n_clusters=5)
kmeans_cluster_5.fit(std_Univ_data)


# In[15]:


labels_5=kmeans_cluster_5.labels_
labels_5


# In[16]:


Univ["cluster_id"]=kmeans_cluster_5.labels_
Univ.head()


# In[17]:


Univ.iloc[:,1:].groupby("cluster_id").agg("mean").reset_index()


# In[18]:


Univ[Univ["cluster_id"]==0]


# #### Silhouette_score

# In[19]:


#Used to define weather a cluster is a good one or not
#if silhouette_score==0 --> clusters are overlapping
#if silhouette_score==-1 --> clusters not homogeneous
from sklearn.metrics import silhouette_score


# In[20]:


help(silhouette_score)


# In[21]:


#silhouette_score when k=3
ss3=silhouette_score(std_Univ_data,labels_3)
ss3


# In[22]:


#silhouette_score when k=5
ss5=silhouette_score(std_Univ_data,labels_5)
ss5


# In[23]:


print(ss3,ss5)


# ## when ss is higher -->> clusters are good
# ### ss3>ss5 -->> 3 clusters is a good option

# In[ ]:




