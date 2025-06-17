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


# In[3]:


# 加载数据
train_data= pd.read_excel("./data/train_data.xlsx",index_col=0)
train_data


# In[4]:


test_data= pd.read_excel("./data/test_data.xlsx",index_col=0)
test_data


# In[5]:


from autogluon.tabular import TabularPredictor
# 加载保存的模型
predictor = TabularPredictor.load('./AutogluonModels/ag-20250605_123204/')


# In[7]:


model_names = predictor.model_names()


# In[8]:


model_names


# In[9]:


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


# In[15]:


# 创建一个空的 DataFrame 用于存储合并后的训练数据与预测值
connected_train_data = train_data.copy()  # 先复制原始数据


# In[16]:


# 遍历每个模型的预测值并添加到 connected_train_data
for model_name, pred_train in predictions_train.items():
    connected_train_data[f'{model_name}'] = pred_train


# In[18]:


# 输出合并后的数据
print("合并后的训练数据：")
print(connected_train_data.head())


# In[19]:


# 合并测试集和预测值
connected_test_data = test_data.copy()
for model_name, pred_test in predictions_test.items():
    connected_test_data[f'{model_name}'] = pred_test


# In[20]:


# 保存为 Excel 文件
connected_train_data.to_excel("./data/train_predict_data.xlsx")
connected_test_data.to_excel("./data/test_predict_data.xlsx")


# In[ ]:




