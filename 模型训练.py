#!/usr/bin/env python
# coding: utf-8

# In[1]:


import autogluon as ag
import pandas as pd
import numpy as np
import os,urllib
import matplotlib.pyplot as plt
from autogluon.tabular import TabularDataset, TabularPredictor
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split


# In[2]:


# 加载数据
train_data= pd.read_excel("train-修改.xlsx",index_col=0)
train_data


# In[3]:


train_data_de_name= train_data.iloc[:,1:]
train_data_de_name


# In[8]:


# 加载数据
test_data= pd.read_excel("test-修改.xls",index_col=0)
test_data


# In[9]:


test_data_de_name= test_data.iloc[:,1:]
test_data_de_name


# In[10]:


label_column = train_data.columns[1]
label_column


# In[11]:


predictor = TabularPredictor(label=label_column,eval_metric="r2",problem_type="regression").fit(train_data_de_name, 
                                                                                                tuning_data=test_data_de_name, 
                                                                                                presets='best_quality',
                                                                                                use_bag_holdout=True,
                                                                                                #path='25-flu-autogluon',
                                                                                                #time_limit = 72000,
                                                                                                auto_stack=True, 
                                                                                                hyperparameters='multimodal',
                                                                                                #feature_prune_kwargs={'prune_threshold': 0.01},
                                                                                                num_bag_folds=5,
                                                                                               )


# In[12]:


#保存最佳模型
predictor.save(silent=True)

predictor.path


# In[14]:


train_true=pd.read_excel("train-.xlsx",index_col=0)
train_true


# In[15]:


test_true=pd.read_excel("test.xls",index_col=0)
test_true


# In[16]:


predictor.leaderboard(test_true)


# In[19]:


predictor.leaderboard(train_true)


# In[20]:


predictor.leaderboard(train_data)


# In[21]:


predictor.leaderboard(test_data)


# In[22]:


validate=pd.read_excel("验证分子.xlsx")


# In[23]:


predictor.predict(validate)


# In[25]:


predictor.predict(train_data).head(10)


# In[26]:


predictor.leaderboard(test_true, extra_metrics=['mae', 'rmse',  'pearsonr'], silent=True)


# In[27]:


predictor.leaderboard(train_true, extra_metrics=['mae', 'rmse',  'pearsonr'], silent=True)


# In[ ]:




