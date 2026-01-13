"""
配置文件：模型超参数和训练配置
"""
import torch

class Config:
    # 数据路径
    TRAIN_FILE_1 = 'data/gaiic_track3_round1_train_20210228.tsv'
    TRAIN_FILE_2 = 'data/gaiic_track3_round2_train_20210407.tsv'
    
    # 模型参数
    VOCAB_SIZE = 34000  # 词表大小，最大词ID是33957
    EMBED_DIM = 300      # 词嵌入维度
    HIDDEN_DIM = 512     # LSTM隐藏层维度
    NUM_LAYERS = 2       # LSTM层数
    DROPOUT = 0.3        # Dropout比例
    MAX_LEN = 60         # 最大序列长度
    
    # 训练参数
    BATCH_SIZE = 256
    LEARNING_RATE = 0.001
    WEIGHT_DECAY = 1e-5
    EPOCHS = 10
    WARMUP_STEPS = 1000
    
    # 其他配置
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    SEED = 42
    NUM_WORKERS = 4
    
    # 验证集划分
    VAL_SPLIT = 0.1
    
    # 模型保存
    MODEL_SAVE_PATH = 'best_model.pth'
    
    def __repr__(self):
        return f"""
Configuration:
--------------
Device: {self.DEVICE}
Vocab Size: {self.VOCAB_SIZE}
Embedding Dim: {self.EMBED_DIM}
Hidden Dim: {self.HIDDEN_DIM}
Batch Size: {self.BATCH_SIZE}
Learning Rate: {self.LEARNING_RATE}
Epochs: {self.EPOCHS}
"""

config = Config()
