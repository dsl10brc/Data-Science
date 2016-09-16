
# coding: utf-8

# In[1]:

#Imports
import numpy as np
import pandas as pd
import sklearn
from sklearn import linear_model
import operator


# In[2]:

columns_name = ["timestamp","placement_id","browser_id","os_id","region","country","is_adserver","campaign",
                "creative_asset_id","mouseovers","clicks","max_duration","video_length","viewable"]


# In[3]:

df_train = pd.read_table('code-test0000_part_00',sep="|",names=columns_name)
# df_test = pd.read_table(test,sep="|",names=columns_name,header=None)


# In[4]:

df_train


# In[5]:

df_train.info()
# df_test.info()


# In[6]:

df_train.drop(["is_adserver","max_duration","video_length"],axis=1,inplace=True)
# df_test.drop(["is_adserver","max_duration","video_length"],axis=1,inplace=True)


# In[7]:

df_train


# In[8]:

df_train = df_train.fillna('-1')
# df_test = df_test.fillna('-1')


# In[9]:

df_train.info()
# df_test.info()


# In[10]:

# Y_train = df_train([])
# X_train = df_train.drop(["clic"])
mask=df_train["country"]=="US"
np.unique(df_train[mask].region)


# In[11]:

tfa = np.vstack(df_train.timestamp.astype(str).apply(lambda x: list(map(str, [x[11:13],x[14:16],x[17:19]]))))
df_train['hours'] = tfa[:,0]
df_train['minute'] = tfa[:,1]
df_train['seconds'] = tfa[:,2]
df_train = df_train.drop(['timestamp'], axis=1)


# In[15]:

df_train = df_train.drop(['timestamp'], axis=1)


# In[16]:

com = list(df_train.columns)


# In[17]:

df_train.info()


# In[18]:

d = ['clicks','hours','minute','seconds']
for i in d:
    com.remove(i)


# In[19]:

df_train = df_train[1:5000]


# In[20]:

sum = 0
for f in com:
    dfg = np.unique(list(df_train[f].values))
    print len(dfg), f


# In[21]:

for i in df_train.columns:
    if (i != 'placement_id') and (i != 'creative_asset_id') and (i != 'browser_id'):
        print pd.DataFrame(pd.value_counts(df_train[i]))


# In[22]:

for f in com:
    df_all_dummy = pd.get_dummies(df_train[f], prefix=f)
    df_train = df_train.drop([f], axis=1)
    df_train = pd.concat((df_train, df_all_dummy), axis=1)


# In[23]:

for dfgs in com:
    affiliate_channel_maxs = []
    affiliate_channel_maxs_dict = {}
    affiliate_channel_maxs = list(enumerate(np.unique(df_train[dfgs])))
    affiliate_channel_maxs_dict = { name : i for i, name in affiliate_channel_maxs }
    df_train[dfgs] = df_train[dfgs].map(lambda x: affiliate_channel_maxs_dict[x]).astype(int)


# In[24]:

df_train


# In[25]:

for i in df_train.columns:
    df_train[i] = df_train[i].astype(int)


# In[26]:

df_train.info()


# In[27]:

# #Greedy add 1 by 1
# scr = []
# c = list(df_train.columns)
# c.remove('clicks')
# for t in range(800):
#     score={}
#     for f in c:
#         if f in scr:
#             pass
#         else:
#             scr.append(f)
#             scr.append('clicks')
#             xz=df_train[scr]
#             xz = xz.dropna()
#             y_t=xz['clicks']
# #             print y_t.shape
#             if len(y_t.index) == 0:
#                 scr.remove(f)
#                 scr.remove('clicks')
#                 continue
#             x_t=xz.drop(['clicks'],axis=1)
# #             print x_t.shape
#             log_model = linear_model.LogisticRegression(penalty='l1',solver='liblinear',multi_class='ovr')
#             #log_model.fit(pd.DataFrame(xz[scr]), pd.DataFrame(xz['Response']))
#             log_model.fit(x_t,y_t)
#             score[f]=log_model.score(x_t,y_t)
#             #score.append([f,log_model.score(pd.DataFrame(xz[scr]),pd.DataFrame(xz['Response']))])
#             scr.remove(f)
#             scr.remove('clicks')
# #     sorted_score = sorted(score.items(), key=operator.itemgetter(1))
# #     print sorted_score
#     scr.append(max(score.iteritems(), key=operator.itemgetter(1))[0])


# In[28]:

# scr


# In[29]:

train_y = df_train['clicks']


# In[30]:

train_x = df_train.drop(['clicks'], axis=1)


# In[33]:

# from xgboost.sklearn import XGBClassifier
# xgg = XGBClassifier(max_depth=6, learning_rate=0.3, n_estimators=25,
#                     objective='binary:logistic', subsample=0.5, colsample_bytree=0.5, seed=0)
from sklearn import cross_validation
from sklearn.ensemble import RandomForestClassifier
forest = RandomForestClassifier(n_estimators=100, min_samples_split=10, random_state=415)
# Compute the accuracy score for all the cross validation folds.  (much simpler than what we did before!)
scores = cross_validation.cross_val_score(forest, train_x, train_y, cv=3)
# Take the mean of the scores (because we have one for each fold)
print(scores.mean())


# In[ ]:



