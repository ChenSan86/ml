"""
一体化脚本：训练 + 评估 基线(BiLSTM) 与 增强版(融合特征工程)

功能：
- 模式 all：依次训练基线与增强版，并在统一 9:1 验证集上评估。
- 模式 eval：跳过训练，直接评估已有权重（best_model.pth 与 best_model_enhanced.pth）。
- 模式 train：仅训练两种模型，不评估（快速调试时使用）。
- 支持 --quick 将 EPOCHS 暂时改为 3。

评估输出：AUC、分类报告、混淆矩阵与预测分布图（由 predict.py 完成）
结果整合：本脚本会将两者 AUC 汇总到 results_baseline_enhanced.csv
"""

import argparse
import csv
from datetime import datetime

from config import config
from dataset import prepare_data
from train import train as train_baseline
from train_enhanced import train_enhanced
from predict import evaluate_model, evaluate_enhanced_model


def write_summary_csv(baseline_auc, enhanced_auc, output_file='results_baseline_enhanced.csv'):
    """写入对比结果到CSV"""
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['timestamp', 'model', 'auc'])
        ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        writer.writerow([ts, 'baseline', f'{baseline_auc:.6f}'])
        writer.writerow([ts, 'enhanced', f'{enhanced_auc:.6f}'])
    print(f"\n💾 结果已写入: {output_file}")


def main():
    parser = argparse.ArgumentParser(description='训练+评估 基线 与 增强版（特征工程）模型')
    parser.add_argument('--mode', type=str, default='all', choices=['train', 'eval', 'all'],
                        help='运行模式：train(训练)、eval(评估现有权重)、all(训练+评估)')
    parser.add_argument('--quick', action='store_true', help='快速模式：将 EPOCHS 临时改为 3')
    parser.add_argument('--baseline_model', type=str, default='best_model.pth',
                        help='基线模型权重路径（评估时使用）')
    parser.add_argument('--enhanced_model', type=str, default='best_model_enhanced.pth',
                        help='增强版模型权重路径（评估时使用）')
    args = parser.parse_args()

    # 快速模式：减小训练周期
    original_epochs = config.EPOCHS
    if args.quick:
        config.EPOCHS = 3
        print(f"⚡ 快速模式已开启：EPOCHS {original_epochs} -> {config.EPOCHS}")

    # 统一准备一次数据划分（9:1），评估时复用
    print("\n📂 准备数据 (9:1 验证集，固定 SEED=42)...")
    train_df, val_df = prepare_data()

    baseline_auc = None
    enhanced_auc = None

    if args.mode in ['train', 'all']:
        print("\n===== 训练：基线模型 =====")
        _model_b, _hist_b = train_baseline()
        print("\n===== 训练：增强版模型（融合特征工程） =====")
        _model_e, _hist_e = train_enhanced()

    if args.mode in ['eval', 'all']:
        print("\n===== 评估：基线模型 =====")
        baseline_auc, _probs_b, _preds_b = evaluate_model(args.baseline_model, val_df)
        print("\n===== 评估：增强版模型 =====")
        enhanced_auc, _probs_e, _preds_e = evaluate_enhanced_model(args.enhanced_model, val_df)

        # 汇总CSV
        write_summary_csv(baseline_auc, enhanced_auc)

    # 还原 EPOCHS 设置
    config.EPOCHS = original_epochs


if __name__ == '__main__':
    main()
