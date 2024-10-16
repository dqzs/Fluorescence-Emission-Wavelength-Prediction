#!/usr/bin/env python
# coding: utf-8

# In[26]:


import autogluon as ag
import pandas as pd
import numpy as np
import os,urllib
import matplotlib.pyplot as plt
from autogluon.tabular import TabularDataset, TabularPredictor
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split


# In[27]:


# 加载数据
train_data= pd.read_excel("train-.xlsx",index_col=0)
train_data


# In[28]:


import pickle
with open('./AutogluonModels/ag-20240829_082340/models/WeightedEnsemble_L2/model.pkl', 'rb') as file:
    predictor = pickle.load(file)


# In[29]:


predictor = TabularPredictor.load('./AutogluonModels/ag-20240829_082340/')


# In[30]:


train_data.iloc[:,1:]


# In[31]:


import shap
import time
shap.initjs()

import warnings
warnings.filterwarnings('ignore')


# In[32]:


test_data= pd.read_excel("test.xls",index_col=0)


# In[33]:


feature_names = train_data.iloc[:,1:].columns
train_data = train_data.copy()

val_data = test_data.iloc[:,1:].copy()

display(train_data)
display(val_data)


# In[34]:


class AutogluonWrapper:
    def __init__(self, predictor, feature_names):
        self.ag_model = predictor
        self.feature_names = feature_names
    
    def predict(self, X):
        if isinstance(X, pd.Series):
            X = X.values.reshape(1,-1)
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=self.feature_names)
        return self.ag_model.predict(X)


# In[17]:


ag_wrapper = AutogluonWrapper(predictor, feature_names)


# In[18]:


# 创建SHAP解释器
explainer = shap.KernelExplainer(ag_wrapper.predict,train_data.iloc[:,1:])


# In[19]:


single_datapoint = train_data.iloc[:,1:]
single_prediction = ag_wrapper.predict(single_datapoint)


# In[37]:


shap_values_single = explainer.shap_values(single_datapoint)


# In[38]:


pd.DataFrame(shap_values_single).to_excel("./shap_values_single_WeightedEnsemble_L2.xlsx",index=False)


# In[ ]:





# In[39]:


pd.read_excel("shap_values_single_WeightedEnsemble_L2.xlsx")


# In[40]:


# 创建数据框
shap_df = pd.DataFrame(shap_values_single[:,1:], columns=train_data.iloc[:,2:].columns.tolist())

# 对每列特征按SHAP值绝对值进行排序
shap_df_abs_mean = shap_df.abs().mean().sort_values(ascending=True)

# 按重要性排序后的特征名称和SHAP值绝对值
feature_names = shap_df_abs_mean.index.tolist()
abs_means = shap_df_abs_mean.values.tolist()

# 创建彩虹渐变色
num_features = len(feature_names)
hues = np.linspace(0, 1, num_features)

# HSV颜色空间中的S和V固定，只变化H（色调）
colors = plt.cm.hsv(hues)

# 生成柱状图
shap.summary_plot(shap_values_single[:,1:], train_data.iloc[:,2:], plot_type='bar', color_bar=True, color=colors, show=False)

# 添加数据值
for i, val in enumerate(abs_means):
    plt.text(val + 0.03, i, round(val, 2))
plt.show()
plt.savefig('shap/shap_bar_WeightedEnsemble_L2.jpg')


# In[41]:


import matplotlib.pyplot as plt
import shap

fig, ax = plt.subplots(figsize=(10, 6))
shap.summary_plot(shap_values_single[:,1:],train_data.iloc[:,2:], color_bar=False,show=False)
# 创建一个ScalarMappable对象
mappable = plt.cm.ScalarMappable(cmap=plt.cm.RdBu)
mappable.set_array([shap_values_single[:,1:].min(), shap_values_single[:,1:].max()])
# 添加Colorbar
cb = plt.colorbar(mappable, ax=ax)
cb.set_label('SHAP train', fontsize=12)
cb.ax.tick_params(labelsize=10)

plt.show()
plt.savefig('shap/shap_values_single_WeightedEnsemble_L2.jpg')


# In[ ]:




