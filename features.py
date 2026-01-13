"""
特征工程模块：提取统计特征和相似度特征
"""
import numpy as np
import torch
from collections import Counter


class FeatureExtractor:
    """文本匹配特征提取器"""
    
    def __init__(self):
        self.feature_names = [
            'len_q1', 'len_q2', 'len_diff', 'len_ratio',
            'word_overlap_count', 'word_overlap_ratio',
            'jaccard_similarity',
            'cosine_similarity',
            'edit_distance_normalized',
            'common_words_ratio_q1', 'common_words_ratio_q2',
            'unique_words_q1', 'unique_words_q2',
            'mean_word_id_q1', 'mean_word_id_q2',
            'max_word_id_q1', 'max_word_id_q2',
            'min_word_id_q1', 'min_word_id_q2'
        ]
    
    def get_feature_dim(self):
        """返回特征维度"""
        return len(self.feature_names)
    
    def extract_features(self, q1_ids, q2_ids):
        """
        提取两个文本的特征
        
        Args:
            q1_ids: 第一个文本的词ID列表（去除padding）
            q2_ids: 第二个文本的词ID列表（去除padding）
        
        Returns:
            features: 特征向量
        """
        # 转为集合
        set1 = set(q1_ids)
        set2 = set(q2_ids)
        
        # 1. 长度特征
        len1 = len(q1_ids)
        len2 = len(q2_ids)
        len_diff = abs(len1 - len2)
        len_ratio = min(len1, len2) / max(len1, len2) if max(len1, len2) > 0 else 0
        
        # 2. 词重叠特征
        common_words = set1 & set2
        overlap_count = len(common_words)
        overlap_ratio = overlap_count / max(len1, len2) if max(len1, len2) > 0 else 0
        
        # 3. Jaccard相似度
        union_words = set1 | set2
        jaccard = len(common_words) / len(union_words) if len(union_words) > 0 else 0
        
        # 4. 余弦相似度（基于词频）
        counter1 = Counter(q1_ids)
        counter2 = Counter(q2_ids)
        cosine = self._cosine_similarity(counter1, counter2)
        
        # 5. 编辑距离（归一化）
        edit_dist = self._levenshtein_distance(q1_ids, q2_ids)
        max_len = max(len1, len2)
        edit_dist_norm = 1 - (edit_dist / max_len) if max_len > 0 else 0
        
        # 6. 公共词占比
        common_ratio_q1 = overlap_count / len1 if len1 > 0 else 0
        common_ratio_q2 = overlap_count / len2 if len2 > 0 else 0
        
        # 7. 唯一词数量
        unique_q1 = len(set1)
        unique_q2 = len(set2)
        
        # 8. 词ID统计特征
        mean_q1 = np.mean(q1_ids) if len(q1_ids) > 0 else 0
        mean_q2 = np.mean(q2_ids) if len(q2_ids) > 0 else 0
        max_q1 = np.max(q1_ids) if len(q1_ids) > 0 else 0
        max_q2 = np.max(q2_ids) if len(q2_ids) > 0 else 0
        min_q1 = np.min(q1_ids) if len(q1_ids) > 0 else 0
        min_q2 = np.min(q2_ids) if len(q2_ids) > 0 else 0
        
        # 组合所有特征
        features = [
            len1, len2, len_diff, len_ratio,
            overlap_count, overlap_ratio,
            jaccard,
            cosine,
            edit_dist_norm,
            common_ratio_q1, common_ratio_q2,
            unique_q1, unique_q2,
            mean_q1, mean_q2,
            max_q1, max_q2,
            min_q1, min_q2
        ]
        
        return np.array(features, dtype=np.float32)
    
    def _cosine_similarity(self, counter1, counter2):
        """计算余弦相似度"""
        # 获取共同的词
        common_words = set(counter1.keys()) & set(counter2.keys())
        
        if not common_words:
            return 0.0
        
        # 计算点积
        dot_product = sum(counter1[word] * counter2[word] for word in common_words)
        
        # 计算模
        norm1 = np.sqrt(sum(count ** 2 for count in counter1.values()))
        norm2 = np.sqrt(sum(count ** 2 for count in counter2.values()))
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)
    
    def _levenshtein_distance(self, seq1, seq2):
        """计算编辑距离（Levenshtein距离）"""
        len1, len2 = len(seq1), len(seq2)
        
        # 创建DP表
        dp = [[0] * (len2 + 1) for _ in range(len1 + 1)]
        
        # 初始化
        for i in range(len1 + 1):
            dp[i][0] = i
        for j in range(len2 + 1):
            dp[0][j] = j
        
        # 填充DP表
        for i in range(1, len1 + 1):
            for j in range(1, len2 + 1):
                if seq1[i-1] == seq2[j-1]:
                    dp[i][j] = dp[i-1][j-1]
                else:
                    dp[i][j] = min(
                        dp[i-1][j] + 1,    # 删除
                        dp[i][j-1] + 1,    # 插入
                        dp[i-1][j-1] + 1   # 替换
                    )
        
        return dp[len1][len2]
    
    def extract_batch_features(self, q1_batch, q2_batch):
        """
        批量提取特征
        
        Args:
            q1_batch: [batch_size, seq_len] tensor
            q2_batch: [batch_size, seq_len] tensor
        
        Returns:
            features: [batch_size, feature_dim] tensor
        """
        batch_size = q1_batch.shape[0]
        feature_dim = self.get_feature_dim()
        features = np.zeros((batch_size, feature_dim), dtype=np.float32)
        
        for i in range(batch_size):
            # 移除padding (假设padding为0)
            q1_ids = q1_batch[i][q1_batch[i] != 0].cpu().numpy().tolist()
            q2_ids = q2_batch[i][q2_batch[i] != 0].cpu().numpy().tolist()
            
            # 提取特征
            features[i] = self.extract_features(q1_ids, q2_ids)
        
        return torch.tensor(features, dtype=torch.float32)


def normalize_features(features, mean=None, std=None):
    """
    特征标准化
    
    Args:
        features: [batch_size, feature_dim] tensor
        mean: 均值，如果为None则计算
        std: 标准差，如果为None则计算
    
    Returns:
        normalized_features, mean, std
    """
    if mean is None:
        mean = features.mean(dim=0, keepdim=True)
    if std is None:
        std = features.std(dim=0, keepdim=True)
        std = torch.where(std == 0, torch.ones_like(std), std)  # 避免除0
    
    normalized = (features - mean) / std
    
    return normalized, mean, std


if __name__ == '__main__':
    # 测试特征提取
    extractor = FeatureExtractor()
    
    # 示例文本
    q1 = [1, 2, 3, 4, 5]
    q2 = [3, 4, 5, 6, 7]
    
    features = extractor.extract_features(q1, q2)
    
    print("特征维度:", len(features))
    print("\n特征名称和值:")
    for name, value in zip(extractor.feature_names, features):
        print(f"  {name:30s}: {value:.4f}")
    
    # 测试批量提取
    q1_batch = torch.tensor([[1, 2, 3, 0, 0], [4, 5, 6, 7, 0]])
    q2_batch = torch.tensor([[2, 3, 4, 0, 0], [5, 6, 7, 8, 9]])
    
    batch_features = extractor.extract_batch_features(q1_batch, q2_batch)
    print(f"\n批量特征形状: {batch_features.shape}")
