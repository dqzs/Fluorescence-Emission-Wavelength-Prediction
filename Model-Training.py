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


# 加载数据集
data= pd.read_excel("data-9.xlsx",index_col=0,)
random_data = data.sample(frac =1).reset_index(drop=True)


# In[125]:


random_data_de_name = random_data.iloc[:,1:]


# In[126]:


random_data_de_name


# In[10]:


# 第二步：分割数据为训练集和测试集
train_data, test_data = train_test_split(random_data_de_name, test_size=0.2, random_state=42) 


# In[10]:


label_column = 'EM'
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


predictor.leaderboard(test_data, extra_metrics=['mae', 'rmse',  'pearsonr'], silent=True)


# In[27]:


predictor.leaderboard(train_data, extra_metrics=['mae', 'rmse',  'pearsonr'], silent=True)


# In[ ]:




