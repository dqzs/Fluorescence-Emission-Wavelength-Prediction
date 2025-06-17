#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from autogluon.tabular import TabularPredictor
import shap
import warnings
from tqdm import tqdm


# In[2]:


# 禁用警告
warnings.filterwarnings('ignore')


# In[3]:


# 创建保存SHAP图片的目录
os.makedirs('./shap_plots', exist_ok=True)


# In[4]:


# 读取数据
train_data = pd.read_excel("./data/train_data.xlsx", index_col=0)
test_data = pd.read_excel("./data/test_data.xlsx", index_col=0)


# In[5]:


# 准备特征数据
feature_names = train_data.drop(columns="EM").iloc[:, 1:].columns
train_features = train_data.drop(columns="EM").iloc[:, 1:].copy()


# In[6]:


# 加载Autogluon predictor
predictor = TabularPredictor.load("./AutogluonModels/ag-20250605_123204")


# In[7]:


train_features


# In[8]:


# 选择需要解释的数据
shap_input = train_features.sample(19838, random_state=42)


# In[9]:


# 明确指定子模型及其对应权重
model_weights = {
    'LightGBMLarge': 0.5,
    'XGBoost': 0.333,
    'LightGBM': 0.083,
    'RandomForestMSE': 0.042,
    'CatBoost': 0.042
}


# In[10]:


# 用于保存所有模型的 SHAP 值
all_shap_values = []


# In[11]:


print("使用指定的模型权重:")
for name, weight in model_weights.items():
    print(f"{name}: {weight:.3f}")

    # 加载每个子模型
    model_obj = predictor._trainer.load_model(name)

    # 仅处理支持 SHAP 的模型
    if hasattr(model_obj, 'model') and hasattr(model_obj.model, 'predict'):
        booster = model_obj.model

        try:
            # 计算SHAP值
            explainer = shap.TreeExplainer(booster)
            shap_vals = explainer.shap_values(shap_input)
            
            # 保存每个模型的SHAP summary plot
            plt.figure(figsize=(12, 8))
            shap.summary_plot(shap_vals, shap_input, feature_names=feature_names)
            plt.title(f"SHAP Summary Plot - {name}")
            plt.savefig(f"./shap_plots/{name}_summary_plot.png", dpi=300, bbox_inches='tight')
            plt.close()

            # 计算并保存每个模型的Mean SHAP条形图
            mean_abs_shap = np.abs(shap_vals).mean(axis=0)
            shap_mean_df = pd.DataFrame({
                'Feature': feature_names,
                'Mean |SHAP|': mean_abs_shap
            }).sort_values(by='Mean |SHAP|', ascending=False)

            plt.figure(figsize=(10, 6))
            bars = plt.barh(shap_mean_df['Feature'], shap_mean_df['Mean |SHAP|'])
            plt.barh(shap_mean_df['Feature'], shap_mean_df['Mean |SHAP|'])
            plt.gca().invert_yaxis()
            plt.xlabel('Mean |SHAP value|')
            plt.title(f'Mean SHAP Values - {name}')
            # 为每个 bar 添加数值标签
            for bar in bars:
                width = bar.get_width()
                plt.text(width + 0.02, bar.get_y() + bar.get_height()/2, f'{width:.4f}', 
                         ha='left', va='center', color='black')

            plt.tight_layout()
            plt.savefig(f"./shap_plots/{name}_mean_shap.png", dpi=300, bbox_inches='tight')
            plt.close()

            # SHAP 值乘以指定权重
            shap_vals_weighted = np.array(shap_vals) * weight
            all_shap_values.append(shap_vals_weighted)
            
            # 保存特征重要性具体数值
            with open(f"./shap_plots/{name}_feature_importance.txt", "w") as f:
                f.write("Feature Importance for model: {}\n\n".format(name))
                for idx, feature in enumerate(shap_mean_df['Feature']):
                    f.write(f"{feature}: {shap_mean_df['Mean |SHAP|'].iloc[idx]:.5f}\n")

        except Exception as e:
            print(f"模型 {name} 无法解释: {e}")


# In[12]:


# 加权平均 SHAP 值
if all_shap_values:
    combined_shap_values = np.sum(all_shap_values, axis=0)

    # 生成加权SHAP summary plot
    plt.figure(figsize=(12, 8))
    shap.summary_plot(combined_shap_values, shap_input, feature_names=feature_names)
    plt.title("Weighted SHAP Summary Plot (Custom Weights)")
    plt.savefig("./shap_plots/weighted_summary_plot.png", dpi=300, bbox_inches='tight')
    plt.close()

    # 生成加权Mean SHAP条形图
    mean_abs_shap_combined = np.abs(combined_shap_values).mean(axis=0)
    shap_mean_combined_df = pd.DataFrame({
        'Feature': feature_names,
        'Mean |SHAP|': mean_abs_shap_combined
    }).sort_values(by='Mean |SHAP|', ascending=False)

    plt.figure(figsize=(10, 6))
    bars = plt.barh(shap_mean_combined_df['Feature'], shap_mean_combined_df['Mean |SHAP|'])
    plt.barh(shap_mean_combined_df['Feature'], shap_mean_combined_df['Mean |SHAP|'])
    plt.gca().invert_yaxis()
    plt.xlabel('Mean |SHAP value|')
    plt.title('Weighted SHAP Feature Importance (Custom Weights)')
    plt.tight_layout()
    # 为每个 bar 添加数值标签
    for bar in bars:
        width = bar.get_width()
        plt.text(width + 0.02, bar.get_y() + bar.get_height()/2, f'{width:.4f}', 
                 ha='left', va='center', color='black')
    plt.savefig("./shap_plots/weighted_mean_shap.png", dpi=300, bbox_inches='tight')
    plt.close()

    # 保存加权特征重要性具体数值
    with open("./shap_plots/weighted_feature_importance.txt", "w") as f:
        f.write("Weighted Feature Importance\n\n")
        for idx, feature in enumerate(shap_mean_combined_df['Feature']):
            f.write(f"{feature}: {shap_mean_combined_df['Mean |SHAP|'].iloc[idx]:.5f}\n")

else:
    print("没有成功解释的模型 SHAP 值。")


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




