"""
主程序入口
"""
import sys
import argparse
from train import train
from predict import evaluate_model
from dataset import prepare_data
from config import config


def main():
    parser = argparse.ArgumentParser(description='文本匹配模型训练和评估')
    parser.add_argument('--mode', type=str, default='train', 
                       choices=['train', 'eval', 'all'],
                       help='运行模式: train(训练), eval(评估), all(训练+评估)')
    parser.add_argument('--model', type=str, default=config.MODEL_SAVE_PATH,
                       help='模型路径（用于评估模式）')
    
    args = parser.parse_args()
    
    if args.mode in ['train', 'all']:
        print("\n" + "=" * 80)
        print("🚀 模式: 训练")
        print("=" * 80)
        model, history = train()
        print("\n✅ 训练完成!")
    
    if args.mode in ['eval', 'all']:
        print("\n" + "=" * 80)
        print("📊 模式: 评估")
        print("=" * 80)
        
        # 准备验证数据
        train_df, val_df = prepare_data()
        
        # 评估模型
        auc, probs, preds = evaluate_model(args.model, val_df)
        
        # 保存预测结果
        from predict import save_predictions
        save_predictions(probs, val_df['label'].values)
        
        print("\n✅ 评估完成!")
    
    print("\n" + "=" * 80)
    print("🎉 所有任务完成!")
    print("=" * 80)


if __name__ == '__main__':
    main()
