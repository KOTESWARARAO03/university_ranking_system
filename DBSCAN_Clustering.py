#!/usr/bin/env python
# coding: utf-8

# In[1]:


#IMPORTING THE LIBRARIES
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler


# In[2]:


df=pd.read_csv("Wholesale customers data.csv")
df.head()


# In[3]:


df.info()


# ### Standardizing the data using StandardScalar

# In[4]:


#convering the data frame into an array
array_data=df.values
array_data


# In[5]:


#standardize the data and store it in std_data
stscaler=StandardScaler()
std_data=stscaler.fit_transform(array_data)


# In[9]:


help(DBSCAN)


# In[16]:


#build and train the model by hyperparameter tuning(changing eps & min_samples values)to get good clustering
#minpts/min_samples >= D+1 (D=no of dimensions)
dbscan=DBSCAN(eps=0.75,min_samples=5)
labels=pd.Series(dbscan.fit_predict(std_data)).value_counts()


# In[17]:


labels


# ### in the above case 131 data points are noise data(more than 30%)

# In[20]:


#build and train the model by hyperparameter tuning(changing eps & min_samples values)to get good clustering
dbscan=DBSCAN(eps=0.9,min_samples=8)
labels=pd.Series(dbscan.fit_predict(std_data)).value_counts()


# In[21]:


#noise is reduced
labels


# In[23]:


dbscan.labels_


# In[24]:


df["cluster"]=dbscan.labels_
df.head()


# In[26]:


#filtering the noisy data
df[df["cluster"]==-1]


# In[27]:


from sklearn.metrics import silhouette_score
silhouette_score(std_data,dbscan.labels_)


# ### higher the silhouette score-->> better the clustering
# #### try out with different eps&min_samples to achieve higher silhouette score

# In[ ]:




