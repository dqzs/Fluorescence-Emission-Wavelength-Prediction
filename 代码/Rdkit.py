
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors
from rdkit.ML.Descriptors import MoleculeDescriptors
from mordred import Calculator, descriptors
import numpy as np
from tqdm import tqdm  # 用于进度条



data =pd.read_excel("./data/EM_Data.xlsx",index_col=0)



# 1. 改进的RDKit描述符计算函数
def calc_rdkit_descriptors(smiles_list):
    # 获取所有可用描述符名称
    desc_names = [desc_name for desc_name, _ in Descriptors._descList]
    calculator = MoleculeDescriptors.MolecularDescriptorCalculator(desc_names)
    
    results = []
    valid_indices = []
    skipped_molecules = []
    
    for idx, smi in tqdm(enumerate(smiles_list), total=len(smiles_list), desc="计算RDKit描述符"):
        try:
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                raise ValueError(f"无效的SMILES: {smi}")
            
            # 添加氢原子以获得更准确的计算
            mol = Chem.AddHs(mol)
            
            # 计算所有描述符
            descriptors = calculator.CalcDescriptors(mol)
            
            # 检查并处理特殊值
            processed_descriptors = []
            for val in descriptors:
                # 处理所有不可用的值类型
                if isinstance(val, float) and (np.isnan(val) or np.isinf(val)):
                    processed_descriptors.append(np.nan)
                elif val is None:  # 处理None值
                    processed_descriptors.append(np.nan)
                else:
                    processed_descriptors.append(val)
            
            results.append(processed_descriptors)
            valid_indices.append(idx)
        except Exception as e:
            skipped_molecules.append((smi, str(e)))
            print(f"跳过SMILES: {smi}, 原因: {str(e)}")
            continue
    
    # 转换为DataFrame并保留有效索引
    df_desc = pd.DataFrame(results, columns=desc_names, index=valid_indices)
    
    # 打印跳过分子的总结
    if skipped_molecules:
        print(f"\n跳过 {len(skipped_molecules)} 个分子:")
        for smi, reason in skipped_molecules:
            print(f"  - {smi}: {reason}")
    
    return df_desc


# In[6]:


# 2. 改进的Mordred描述符计算函数（不再需要Missing）
def calc_mordred_descriptors(smiles_list):
    # 创建仅包含2D描述符的计算器
    calc = Calculator(descriptors, ignore_3D=True)
    
    results = []
    valid_smiles = []
    skipped_molecules = []
    
    for smi in tqdm(smiles_list, desc="计算Mordred描述符"):
        try:
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                raise ValueError(f"无效的SMILES: {smi}")
            
            # 添加氢原子以获得更准确的计算
            mol = Chem.AddHs(mol)
            
            # 计算描述符
            res = calc(mol)
            
            # 处理结果，保留原始值或转换为NaN
            descriptor_dict = {}
            for key, val in res.asdict().items():
                # 处理NaN和无穷大
                if isinstance(val, float) and (np.isnan(val) or np.isinf(val)):
                    descriptor_dict[key] = np.nan
                # 处理None值
                elif val is None:
                    descriptor_dict[key] = np.nan
                # 处理Mordred的Missing值（无需特殊导入）
                elif hasattr(val, '__class__') and val.__class__.__name__ == 'Missing':  # 替代Missing检查
                    descriptor_dict[key] = np.nan
                else:
                    descriptor_dict[key] = val
            
            results.append(descriptor_dict)
            valid_smiles.append(smi)
        except Exception as e:
            skipped_molecules.append((smi, str(e)))
            print(f"跳过SMILES: {smi}, 原因: {str(e)}")
            continue
    
    # 创建DataFrame
    df_mordred = pd.DataFrame(results)
    df_mordred['SMILES'] = valid_smiles
    
    # 打印跳过分子的总结
    if skipped_molecules:
        print(f"\n跳过 {len(skipped_molecules)} 个分子:")
        for smi, reason in skipped_molecules:
            print(f"  - {smi}: {reason}")
    
    return df_mordred


# In[7]:


# 3. 改进的特征合并函数
def merge_features_without_duplicates(original_df, *feature_dfs):
    """合并多个特征DataFrame并去除重复列"""
    # 按顺序合并（后出现的重复列会被丢弃）
    merged = pd.concat([original_df] + list(feature_dfs), axis=1)
    
    # 保留第一个出现的列（根据合并顺序）
    merged = merged.loc[:, ~merged.columns.duplicated()]
    
    # 添加特征统计信息
    print("\n特征统计:")
    print(f"总特征数: {len(merged.columns)}")
    
    # 计算缺失值比例
    missing_percentage = merged.isna().mean().sort_values(ascending=False)
    high_missing = missing_percentage[missing_percentage > 0.5]
    
    if not high_missing.empty:
        print("\n高缺失值特征 (>50%):")
        for feature, percentage in high_missing.items():
            print(f"  - {feature}: {percentage:.1%}")
    
    return merged


# In[8]:


# 计算描述符
print("开始计算RDKit描述符...")
rdkit_features = calc_rdkit_descriptors(data['SMILES'])


# In[9]:


print("\n开始计算Mordred描述符...")
mordred_features = calc_mordred_descriptors(data['SMILES'])


# In[10]:


# 对齐有效行
# 获取两个特征集的交集索引
common_indices = rdkit_features.index.intersection(mordred_features.index)


# In[11]:


if len(common_indices) < len(data):
    print(f"警告: 只有 {len(common_indices)}/{len(data)} 个分子被成功处理")


# In[12]:


# 确保所有DataFrame使用相同的索引
valid_df = data.iloc[common_indices].reset_index(drop=True)
rdkit_clean = rdkit_features.loc[common_indices].reset_index(drop=True)
mordred_clean = mordred_features.loc[common_indices].drop('SMILES', axis=1).reset_index(drop=True)


# In[13]:


# 智能合并并去重
print("\n合并特征...")
final_df = merge_features_without_duplicates(valid_df, rdkit_clean, mordred_clean)
final_df


# In[14]:


final_df.to_csv("./data/EM_all_feature.csv")


# In[ ]:




