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


table_data = pd.read_excel("./data/6_feature_selection.xlsx",index_col=0)
train_data,test_data = train_test_split(table_data,test_size=0.2,random_state=42)


# In[4]:


train_data.to_excel("./data/train_data.xlsx")
train_data


# In[5]:


test_data.to_excel("./data/test_data.xlsx")
test_data


# In[7]:


test_data.iloc[:,1:]


# In[25]:


predictor = TabularPredictor(label="EM",eval_metric="r2",problem_type="regression").fit(train_data.iloc[:,1:], 
                                                                                                tuning_data=test_data.iloc[:,1:],
                                                                                                #presets='best_quality',
                                                                                                #use_bag_holdout=True
                                                                                               )


# In[26]:


#保存最佳模型
predictor.save(silent=True)


# In[27]:


predictor.model_best


# In[28]:


predictor.path


# In[29]:


predictor.leaderboard(train_data,extra_metrics=['mae', 'rmse',  'pearsonr'])


# In[30]:


predictor.leaderboard(test_data,extra_metrics=['mae', 'rmse',  'pearsonr'])


# In[31]:


predictor.model_names()


# In[33]:


feature_importance = predictor.feature_importance(train_data)


# In[34]:


feature_importance.to_excel("./data/6_feature_import.xlsx")


# In[35]:


feature_importance







