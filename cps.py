
# coding: utf-8

# In[1]:

import pandas as pd
pd.set_option('max_columns',72)


# In[2]:

import numpy as np


# In[3]:

def print_full(x):
    pd.set_option('display.max_rows', len(x))
    print(x)
    pd.reset_option('display.max_rows')


# In[4]:

colu = pd.read_excel('Data Dictionary.xls',sheetname=0)
listcol = list(colu.Name)


# In[5]:

datas= pd.read_csv('finaldat.csv')


# In[6]:

datas


# In[7]:

date = ["_DT","_TS","_TM"]


# In[8]:

tate = []
for i in listcol:
    if (i[-3:] in date):
        tate.append(i)


# In[9]:

for col in tate:
        del datas[col]


# In[10]:

cate=[]
s = ["_CD","_ID","IND","OME"]
for i in listcol:
    if (i[-3:] in s) & (i!='AUTH_ID'):
        cate.append(i)


# In[11]:

cate


# In[12]:

for dfgs in cate:
    affiliate_channel_maxs = []
    affiliate_channel_maxs_dict = {}
    affiliate_channel_maxs = list(enumerate(np.unique(datas[dfgs])))
    affiliate_channel_maxs_dict = { name : i for i, name in affiliate_channel_maxs }
    datas[dfgs] = datas[dfgs].map(lambda x: affiliate_channel_maxs_dict[x]).astype(int)


# In[13]:

datas = datas.drop(['ACCT_ID_TOKEN'], axis=1)


# In[14]:

datas = datas.drop(['AUTH_ID'], axis=1)


# In[15]:

backcoh = ['AUTHZN_TRMNL_PIN_CAPBLT_NUM']
for dfgs in backcoh:
    affiliate_channel_maxs = []
    affiliate_channel_maxs_dict = {}
    affiliate_channel_maxs = list(enumerate(np.unique(datas[dfgs])))
    affiliate_channel_maxs_dict = { name : i for i, name in affiliate_channel_maxs }
    datas[dfgs] = datas[dfgs].map(lambda x: affiliate_channel_maxs_dict[x]).astype(int)


# In[16]:

for i in datas.columns:
    datas[i] = datas[i].astype(int)


# In[17]:

datas.info(verbose=True, null_counts=True)


# In[18]:

train_y = datas['FRD_IND']
train_x = datas.drop(['FRD_IND','Unnamed: 0','Unnamed: 0.1'], axis=1)


# In[ ]:

from sklearn import cross_validation
from sklearn.ensemble import RandomForestClassifier
from scipy.stats import randint as sp_randint
from sklearn.grid_search import RandomizedSearchCV
forest = RandomForestClassifier(n_estimators=100)
param_dist = {"max_depth": [3, None],
              "max_features": sp_randint(1, 11),
              "min_samples_split": sp_randint(1, 11),
              "min_samples_leaf": sp_randint(1, 11),
              "bootstrap": [True, False],
              "criterion": ["gini", "entropy"]}

# run randomized search
n_iter_search = 20
random_search = RandomizedSearchCV(forest, param_distributions=param_dist,
                                   n_iter=n_iter_search)
random_search.fit(train_x, train_y)
# scores = cross_validation.cross_val_score(forest, train_x, train_y, cv=3)
# print(scores.mean())


# In[ ]:



