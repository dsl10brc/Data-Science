
# coding: utf-8

# In[2]:

import pandas as pd
dats = pd.read_csv('26.csv')


# In[3]:

pd.set_option('max_columns',75)


# In[4]:

dats


# In[5]:

from sklearn.cluster import KMeans


# In[6]:

df = dats.dropna(axis=1,how='all')


# In[7]:

df


# In[8]:

df.info()


# In[9]:

fd = df.dropna()


# In[10]:

fd


# In[11]:

# pd.set_option('max_rows',10000000)
inds = pd.isnull(df).any(1).nonzero()[0]


# In[12]:

def print_full(x):
    pd.set_option('display.max_rows', len(x))
    print(x)
    pd.reset_option('display.max_rows')


# In[13]:

import numpy as np
np.set_printoptions(threshold='nan')
inds


# In[17]:

inds = pd.DataFrame(inds)


# In[19]:

file_ = 'vbn'
fg = str(file_)


# In[20]:

inds.to_csv(fg+'bv.csv')


# In[31]:

fdn = df[df.isnull().any(axis=1)]


# In[32]:

fdn


# In[33]:

import math
d1 = len(fdn)
d2 = len(df)
d3 = d1/float(d2)
print d3*100,'% of missing data'


# In[34]:

fdcol = fd.columns[0:4]


# In[35]:

for col in fdcol:
    del fd[col]


# In[36]:

fd


# In[37]:

fdf = fd.drop(['use'],axis=1)


# In[38]:

fdf1 = fdf.transpose()


# In[39]:

fdf1


# In[40]:

k_mea= KMeans(init='k-means++', n_clusters=15)
# k_means.fit(X)


# In[41]:

barc = k_mea.fit(fdf1)


# In[42]:

fdf2 = pd.DataFrame(barc.cluster_centers_)


# In[44]:

fdf5 = pd.DataFrame(barc.labels_)


# In[45]:

fdf5.transpose()


# In[43]:

fdf2.transpose()


# In[ ]:

k_mean= KMeans(init='k-means++', n_clusters=10)
bac = k_mean.fit(fd)


# In[ ]:

centroids= k_mean.cluster_centers_


# In[ ]:

labels = k_mean.labels_


# In[ ]:

pd.DataFrame(centroids)


# In[ ]:

pd.DataFrame(labels)


# In[ ]:

ghas = KMeans(init='k-means++', n_clusters=14)
garc = k_mea.fit_transform(fd)


# In[ ]:

# import matplotlib.pyplot as plt

# # Make a histogram of all the ratings in the average_rating column.
# plt.hist((fd2["use"]))

# # Show the plot.
# plt.show()



