#!/usr/bin/env python
# coding: utf-8

# In[1]:


import rdkit
import pandas as pd
import numpy as np
from collections import Counter

from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.ML.Descriptors import MoleculeDescriptors
from mordred import Calculator, descriptors
from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)


# In[2]:


# 尝试使用不同的编码方式
try_encodings = ['utf-8', 'gbk', 'latin1', 'ISO-8859-1']

for encoding in try_encodings:
    try:
        final_data_1 = pd.read_csv("./data/EM_all_feature.csv", index_col=0, encoding=encoding)
        print(f"成功读取文件，编码方式为：{encoding}")
        break
    except UnicodeDecodeError:
        continue


# In[3]:


final_data_1


# In[4]:


final_data = final_data_1.iloc[:,9:]
final_data


# In[5]:


em = final_data_1.iloc[:,2]
em


# In[6]:


final_data.insert(0,"EM",em)


# In[7]:


final_data


# In[ ]:





# In[8]:


# 删除0值和空值
consolidated_data = final_data.apply(lambda x: pd.to_numeric(x, errors='coerce'))
consolidated_data  = consolidated_data.dropna(axis=1,how='any')
print(consolidated_data.shape)
data0=(consolidated_data==0).sum(axis = 0)
number0 = data0[data0>=final_data.shape[0]/2]
number0.index
consolidated_data = consolidated_data.drop(labels=number0.index,axis=1)
print(consolidated_data.shape)
consolidated_data.head()


# In[9]:


# 删除低方差的值
var_data = consolidated_data.iloc[:,3:].var()
del_col = var_data[var_data<0.1].index
consolidated_data = consolidated_data.drop(labels=del_col,axis=1)
print(consolidated_data.shape)
consolidated_data.head()


# In[10]:


correlation = consolidated_data.iloc[:,:].corr('spearman')
correlation


# In[11]:


df_bool = (abs(correlation) > 0.85)
correlation[df_bool]


# In[12]:


DN_correlation = consolidated_data.iloc[:,:].corr('spearman')["EM"]
DN_correlation[DN_correlation>0.1]


# In[13]:


bb = []
col_index = correlation.index
for i in range(0,len(col_index)):
    for j in range(i+1,len(col_index)):
        bb.append([col_index[i],col_index[j]])


# In[14]:


import math
k = 0
del_list = []
for i in bb:
    if not math.isnan(correlation[df_bool].loc[i[0],i[1]]) and ('EM'not in i):
        k+=1
        if abs(DN_correlation[i[0]])>abs(DN_correlation[i[1]]):
            if i[1] not in del_list:
                del_list.append(i[1])
        else:
            if i[0] not in del_list:
                del_list.append(i[0])
            
print(del_list)
k


# In[15]:


Test_data = consolidated_data.drop(labels=del_list,axis=1)
Test_data


# In[16]:


data = final_data_1.iloc[:,:9]
data


# In[17]:


Test_data


# In[18]:


result =pd.concat([data, Test_data], axis=1)


# In[20]:


result.to_csv("./data/EM_data_processing.csv")


# In[ ]:




