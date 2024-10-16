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


# In[6]:


# 加载数据
train_data= pd.read_excel("train.xlsx",index_col=0)
train_data


# In[7]:


test_data= pd.read_excel("test.xls",index_col=0)
test_data


# In[8]:


from autogluon.tabular import TabularPredictor
# 加载保存的模型
predictor = TabularPredictor.load('./AutogluonModels/ag-20240829_082340/')


# In[9]:


model_names = predictor.get_model_names()


# In[10]:


from autogluon.tabular import TabularPredictor

# 假设 predictor2 是你已经训练好的 TabularPredictor 对象
# model_names 是通过 predictor2.get_model_names() 获取的模型名称列表
# train_data 是你的训练数据集
# test_data 是你的测试数据集

predictions_train = {}
predictions_test = {}

for model_name in model_names:
    # 使用特定模型在训练集上进行预测
    pred_train = predictor.predict(train_data, model=model_name)
    predictions_train[model_name] = pred_train
    
    # 使用特定模型在测试集上进行预测
    pred_test = predictor.predict(test_data, model=model_name)
    predictions_test[model_name] = pred_test


# In[11]:


pd.DataFrame(predictions_test).to_excel("predictions_test.xlsx")
pd.DataFrame(predictions_train).to_excel("predictions_train.xlsx")


# ## 验证分子

# In[39]:


# 加载数据
data= pd.read_excel("./验证分子.xlsx",index_col=0)
data


# In[40]:


from autogluon.tabular import TabularPredictor

# 假设 predictor2 是你已经训练好的 TabularPredictor 对象
# model_names 是通过 predictor2.get_model_names() 获取的模型名称列表
# train_data 是你的训练数据集
# test_data 是你的测试数据集

predictions_data = {}

for model_name in model_names:
    # 使用特定模型在训练集上进行预测
    pred_data = predictor.predict(data, model=model_name)
    predictions_data[model_name] = pred_data



# In[41]:


predictions_data


# In[42]:


pd.DataFrame(predictions_data).to_excel("predictions_验证.xlsx")


# In[ ]:




