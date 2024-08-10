#!/usr/bin/env python
# coding: utf-8

# In[1]:


# import hierarchical clustering libraries
import scipy.cluster.hierarchy as sch  #for dendrogram
from sklearn.cluster import AgglomerativeClustering
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sn


# In[2]:


Univ=pd.read_csv("Universities.csv")
Univ


# ### step1-Standardization

# In[3]:


#min-max scaling for categorical variables
def norm_fun(i):
    x=(i-i.min())/(i.max()-i.min())
    return (x)


# In[4]:


#standard scaling for continuous variables
def std_fun(i):
    x=(i-i.mean())/i.std()
    return (x)


# In[5]:


#standardized data frame (we have considered all the numerical cols only)
df_std=std_fun(Univ.iloc[:,1:])


# In[6]:


Univ.iloc[:,1:].describe()


# In[7]:


df_std.describe()


# ### Step2-Calculate the distances

# In[8]:


help(sch.linkage)


# In[9]:


#distance metrics that can be used to calculate pairwise distances
from scipy.spatial.distance import pdist
help(pdist)


# In[10]:


#calculating the distance matrix
dist_mat=sch.linkage(df_std,metric='euclidean', method='complete')
dist_mat


# ### Step3-create dendrogram

# In[11]:



# x-axis-->indexes of the observations
# y_axis-->distances between them
dendrogram = sch.dendrogram(dist_mat)


# ### Step4-Agglomarative Clustering

# In[12]:


# create clusters
hc = AgglomerativeClustering(n_clusters=3, affinity = 'euclidean', linkage = 'complete')
hc.fit(df_std)


# ### Step-5 Cluster Analysis

# In[13]:


#tells us about the cluster label ie which data point belongs to which to cluster label
hc.labels_


# In[14]:


Univ["cluster_id"]=hc.labels_


# In[15]:


Univ.head()


# In[16]:


#sorting the univerties based on the cluster id
Univ1 = Univ.sort_values("cluster_id")
Univ1.iloc[:,[0,-1]]


# In[17]:


#comparing the universities and making inferences based on their cluster id
Univ.iloc[:,1:].groupby("cluster_id").mean()


# ### Inferences made
# #### Tier1 Universities-Univ in cluster 1
# #### Tier2 Universities-Univ in cluster 2
# #### Tier3 Universities-Univ in cluster 0
# 

# ### applying for all colleges needs some money and  also meet the needs and requirements of those colleges. 
# ### If we can categorize them based on the key attributes like expenses,gradrate then we can help the 
# ### The students in choosing the best fit colleges  that they can apply and surely get seat in it

# ## Inferences-
# ### if your SATscore is high-Tier1
# ### if your SATscore is good and want to spend less expenses-Tier2
# ### if your SATscore is low -Tier3
# ### here we presented only 25 univ only,but if you have hundreds of univercities in your list then this analysis is much useful

# ### for univ management,you can see who are your competitors and what are their attributes.Helps the management to grow 

# ### if we increase the size of clusters we can make much more accurate inferences. 

# In[ ]:




