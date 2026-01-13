"""
数据加载和预处理模块
"""
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from config import config


def load_data(file_paths):
    """加载并合并多个数据文件"""
    dfs = []
    for file_path in file_paths:
        df = pd.read_csv(file_path, sep='\t', header=None, 
                        names=['query1', 'query2', 'label'])
        dfs.append(df)
    
    combined_df = pd.concat(dfs, ignore_index=True)
    print(f"✅ 加载数据完成: {len(combined_df):,} 条样本")
    print(f"   正样本: {(combined_df['label']==1).sum():,} ({combined_df['label'].mean():.2%})")
    print(f"   负样本: {(combined_df['label']==0).sum():,}")
    
    return combined_df


def text_to_ids(text, max_len=60):
    """将文本转换为ID序列"""
    ids = [int(x) for x in str(text).split()]
    
    # 截断
    if len(ids) > max_len:
        ids = ids[:max_len]
    
    # padding
    padding_len = max_len - len(ids)
    ids = ids + [0] * padding_len
    
    return ids


class TextMatchDataset(Dataset):
    """文本匹配数据集"""
    
    def __init__(self, df, max_len=60):
        self.query1 = df['query1'].values
        self.query2 = df['query2'].values
        self.labels = df['label'].values
        self.max_len = max_len
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        q1_ids = text_to_ids(self.query1[idx], self.max_len)
        q2_ids = text_to_ids(self.query2[idx], self.max_len)
        label = self.labels[idx]
        
        return {
            'query1': torch.tensor(q1_ids, dtype=torch.long),
            'query2': torch.tensor(q2_ids, dtype=torch.long),
            'label': torch.tensor(label, dtype=torch.float)
        }


def get_dataloaders(train_df, val_df, batch_size=256, num_workers=4):
    """创建训练和验证数据加载器"""
    train_dataset = TextMatchDataset(train_df, max_len=config.MAX_LEN)
    val_dataset = TextMatchDataset(val_df, max_len=config.MAX_LEN)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader


def prepare_data():
    """准备训练和验证数据"""
    # 加载数据
    df = load_data([config.TRAIN_FILE_1, config.TRAIN_FILE_2])
    
    # 划分训练集和验证集
    train_df, val_df = train_test_split(
        df, 
        test_size=config.VAL_SPLIT,
        random_state=config.SEED,
        stratify=df['label']  # 保持标签分布一致
    )
    
    print(f"\n📊 数据划分:")
    print(f"   训练集: {len(train_df):,} 条")
    print(f"   验证集: {len(val_df):,} 条")
    
    return train_df, val_df


if __name__ == '__main__':
    # 测试数据加载
    train_df, val_df = prepare_data()
    train_loader, val_loader = get_dataloaders(train_df, val_df, batch_size=4)
    
    # 打印一个batch
    batch = next(iter(train_loader))
    print(f"\n🔍 Batch示例:")
    print(f"   Query1 shape: {batch['query1'].shape}")
    print(f"   Query2 shape: {batch['query2'].shape}")
    print(f"   Labels shape: {batch['label'].shape}")
    print(f"   Query1[0]: {batch['query1'][0][:10]}...")
    print(f"   Label[0]: {batch['label'][0]}")
