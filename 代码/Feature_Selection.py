#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import pandas as pd


# In[3]:


table_data = pd.read_csv("./data/EM_data_processing.csv",index_col=0)
table_data


# In[4]:


# 删除指定的列：Solvent 和 DOI
columns_to_drop = ['Solvent', 'DOI']


# In[7]:


table_data =  table_data.drop(columns=columns_to_drop,errors="ignore")
table_data


# In[9]:


table_data.iloc[:,2:]


# In[11]:


import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'  # 使用国内镜像

from autogluon.tabular import TabularPredictor
predictor = TabularPredictor(label='EM',eval_metric="r2").fit(
                                                               train_data=table_data.iloc[:,2:],
                                                               #presets='best_quality',
                                                               #hyperparameters=hyperparameters,
                                                               #feature_metadata=feature_metadata,
                                
)


# In[12]:


predictor.model_best


# In[13]:


predictor.path


# In[14]:


table_data.iloc[:,7:]


# In[15]:


predictor.leaderboard(table_data)


# In[ ]:





# In[17]:


feature_importance = predictor.feature_importance(table_data.iloc[:,2:])
feature_importance


# In[18]:


feature_importance.to_excel("data/feature_import.xlsx")


# In[19]:


feature_importance = pd.read_excel("data/feature_import.xlsx",index_col=0)
feature_importance


# In[20]:


feature_importance = feature_importance.drop(["Et30", "SP", "SdP", "SA", "SB"], axis=0)
feature_importance


# In[28]:


features= feature_importance.index[:2].tolist()
features


# In[30]:


table_data["nBondsD"]


# In[26]:


table_data.iloc[:,2:8]


# In[ ]:





# In[29]:


df1 = pd.DataFrame(table_data.iloc[:,2:8])
df2=pd.DataFrame(table_data[features])
data_1= pd.concat([df1, df2],axis=1)
data_1


# In[32]:


# 根据特征重要性的排名重新训练数据，并绘制得分图
scores = []
for i in range(1, 127):
    # 选择前i个特征
    features = feature_importance.index[:i].tolist()
    #数据集进行链接
    df1 = pd.DataFrame(table_data.iloc[:,2:8])
    df2=pd.DataFrame(table_data[features])
    data_1= pd.concat([df1, df2],axis=1)
    data_1
    # 训练模型
    predictor = TabularPredictor(label="EM",eval_metric="r2").fit(data_1,
                                                                          #hyperparameters='multimodal',
                                                                          #presets='good_quality',
                                                                         )
    
    # 计算得分
    score = predictor.evaluate(data_1)["r2"]
    
    #y_pred = predictor.predict(test_data)
    #score = r2_score(test_data[label_column], y_pred)
    scores.append(score)
    
    # 显示进度
    print(f'Training with top {i} features, score = {score:.4f}')


# In[33]:


from pylab import xticks,yticks,np
# 绘制得分图
import matplotlib.pyplot as plt
# 查看前20个特征的得分
plt.figure(figsize=(11,4))
plt.title("Features-R2")
#  选择的特征数量
plt.xlabel("Number of features selected")
# 交叉验证得分
plt.ylabel("R2_Score")

# 修改横轴坐标刻度
xticks(np.linspace(1,21,21,endpoint=True))
# yticks(np.linspace(0.7,1,7,endpoint=True))

# 画出各个特征的得分zzzzz
plt.plot(range(1, len(scores[:20])+1),scores[:20])
plt.grid()
plt.show()


# In[34]:


df = pd.DataFrame(scores)


# In[35]:


df.to_excel("./data/scores_127.xlsx")


# In[36]:


# 选择前i个特征
features = feature_importance.index[:6].tolist()
features


# In[37]:


df1 = pd.DataFrame(table_data.iloc[:,:8])
df2=pd.DataFrame(table_data[features])
data_1= pd.concat([df1, df2],axis=1)
data_1


# In[38]:


data_1.to_excel("./data/6_feature_selection.xlsx")


# In[ ]:




