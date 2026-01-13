"""
消融实验主脚本：系统化评估每个改进方向的效果
"""
import os
import json
import time
import torch
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from tqdm import tqdm

from config import config
from dataset import prepare_data, get_dataloaders
from train import set_seed, compute_metrics, train_epoch, evaluate

# 导入不同的模型
from model import create_model as create_baseline_model
from model_attention import create_attention_model
from model_enhanced import create_enhanced_model


class AblationStudy:
    """消融实验管理器"""
    
    def __init__(self, output_dir='ablation_results'):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        self.results = []
        self.experiment_log = []
        
        # 设置随机种子
        set_seed(config.SEED)
        
        # 准备数据（只加载一次）
        print("\n" + "="*80)
        print("📂 准备数据...")
        print("="*80)
        self.train_df, self.val_df = prepare_data()
        self.train_loader, self.val_loader = get_dataloaders(
            self.train_df, self.val_df,
            batch_size=config.BATCH_SIZE,
            num_workers=config.NUM_WORKERS
        )
        
        print(f"\n✅ 数据准备完成")
        print(f"   训练集: {len(self.train_df):,} 条")
        print(f"   验证集: {len(self.val_df):,} 条")
    
    def train_single_experiment(self, model, model_name, epochs=5):
        """
        训练单个实验
        
        Args:
            model: 模型实例
            model_name: 模型名称
            epochs: 训练轮数
        """
        print("\n" + "="*80)
        print(f"🚀 实验: {model_name}")
        print("="*80)
        
        model = model.to(config.DEVICE)
        
        # 损失函数和优化器
        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.LEARNING_RATE,
            weight_decay=config.WEIGHT_DECAY
        )
        
        # 学习率调度器
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs, eta_min=1e-6
        )
        
        # 记录
        history = {
            'train_loss': [],
            'train_auc': [],
            'val_loss': [],
            'val_auc': []
        }
        
        best_auc = 0.0
        start_time = time.time()
        
        # 训练循环
        for epoch in range(1, epochs + 1):
            print(f"\n--- Epoch {epoch}/{epochs} ---")
            
            # 训练
            train_loss, train_auc = train_epoch(
                model, self.train_loader, criterion, optimizer, config.DEVICE
            )
            
            # 验证
            val_loss, val_metrics = evaluate(
                model, self.val_loader, criterion, config.DEVICE
            )
            
            # 更新学习率
            scheduler.step()
            
            # 记录
            history['train_loss'].append(train_loss)
            history['train_auc'].append(train_auc)
            history['val_loss'].append(val_loss)
            history['val_auc'].append(val_metrics['auc'])
            
            # 更新最佳
            if val_metrics['auc'] > best_auc:
                best_auc = val_metrics['auc']
            
            print(f"Train Loss: {train_loss:.4f}, Train AUC: {train_auc:.4f}")
            print(f"Val Loss: {val_loss:.4f}, Val AUC: {val_metrics['auc']:.4f}")
        
        total_time = time.time() - start_time
        
        # 保存结果
        result = {
            'model_name': model_name,
            'best_val_auc': best_auc,
            'final_val_auc': val_metrics['auc'],
            'final_accuracy': val_metrics['accuracy'],
            'final_precision': val_metrics['precision'],
            'final_recall': val_metrics['recall'],
            'final_f1': val_metrics['f1'],
            'training_time': total_time,
            'epochs': epochs,
            'history': history
        }
        
        self.results.append(result)
        
        print(f"\n✅ {model_name} 完成!")
        print(f"   Best AUC: {best_auc:.4f}")
        print(f"   Training Time: {total_time:.1f}s")
        
        # 清理内存
        del model, optimizer, scheduler
        torch.cuda.empty_cache()
        
        return result
    
    def run_baseline_experiment(self, epochs=5):
        """实验1: 基线模型（BiLSTM双塔）"""
        print("\n" + "="*80)
        print("📊 实验 1/10: 基线模型（BiLSTM双塔）")
        print("="*80)
        
        model = create_baseline_model()
        result = self.train_single_experiment(model, "1_Baseline_BiLSTM", epochs)
        
        self.log_experiment(
            experiment_id=1,
            name="Baseline BiLSTM",
            description="双塔BiLSTM + 交互层",
            improvements=[]
        )
        
        return result
    
    def run_attention_experiment(self, epochs=5):
        """实验2: 基线 + 注意力机制"""
        print("\n" + "="*80)
        print("📊 实验 2/10: 基线 + 注意力机制")
        print("="*80)
        
        model = create_attention_model()
        result = self.train_single_experiment(model, "2_Baseline+Attention", epochs)
        
        self.log_experiment(
            experiment_id=2,
            name="Baseline + Attention",
            description="在BiLSTM基础上添加自注意力机制",
            improvements=["Attention Mechanism"]
        )
        
        return result
    
    def run_feature_engineering_experiment(self, epochs=5):
        """实验3: 基线 + 特征工程"""
        print("\n" + "="*80)
        print("📊 实验 3/10: 基线 + 特征工程（19个手工特征）")
        print("="*80)
        
        model = create_enhanced_model()
        result = self.train_single_experiment(model, "3_Baseline+Features", epochs)
        
        self.log_experiment(
            experiment_id=3,
            name="Baseline + Feature Engineering",
            description="添加19个手工特征（长度、重叠、相似度等）",
            improvements=["19 Handcrafted Features"]
        )
        
        return result
    
    def run_focal_loss_experiment(self, epochs=5):
        """实验4: 基线 + Focal Loss"""
        print("\n" + "="*80)
        print("📊 实验 4/10: 基线 + Focal Loss（难样本挖掘）")
        print("="*80)
        
        from losses import FocalLoss
        
        model = create_baseline_model()
        model = model.to(config.DEVICE)
        
        # 使用Focal Loss
        criterion = FocalLoss(alpha=0.25, gamma=2.0)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.LEARNING_RATE,
            weight_decay=config.WEIGHT_DECAY
        )
        
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs, eta_min=1e-6
        )
        
        best_auc = 0.0
        start_time = time.time()
        
        for epoch in range(1, epochs + 1):
            print(f"\n--- Epoch {epoch}/{epochs} ---")
            
            # 训练
            model.train()
            total_loss = 0
            all_labels, all_probs = [], []
            
            for batch in tqdm(self.train_loader, desc='Training'):
                query1 = batch['query1'].to(config.DEVICE)
                query2 = batch['query2'].to(config.DEVICE)
                labels = batch['label'].to(config.DEVICE)
                
                optimizer.zero_grad()
                logits = model(query1, query2)
                loss = criterion(logits, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                
                total_loss += loss.item()
                probs = torch.sigmoid(logits).detach().cpu().numpy()
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs)
            
            from sklearn.metrics import roc_auc_score
            train_loss = total_loss / len(self.train_loader)
            train_auc = roc_auc_score(all_labels, all_probs)
            
            # 验证（使用BCE评估）
            val_loss, val_metrics = evaluate(
                model, self.val_loader, nn.BCEWithLogitsLoss(), config.DEVICE
            )
            
            scheduler.step()
            
            if val_metrics['auc'] > best_auc:
                best_auc = val_metrics['auc']
            
            print(f"Train Loss: {train_loss:.4f}, Train AUC: {train_auc:.4f}")
            print(f"Val Loss: {val_loss:.4f}, Val AUC: {val_metrics['auc']:.4f}")
        
        total_time = time.time() - start_time
        
        result = {
            'model_name': "4_Baseline+FocalLoss",
            'best_val_auc': best_auc,
            'final_val_auc': val_metrics['auc'],
            'final_accuracy': val_metrics['accuracy'],
            'final_precision': val_metrics['precision'],
            'final_recall': val_metrics['recall'],
            'final_f1': val_metrics['f1'],
            'training_time': total_time,
            'epochs': epochs
        }
        
        self.results.append(result)
        self.log_experiment(
            experiment_id=4,
            name="Baseline + Focal Loss",
            description="使用Focal Loss处理难样本",
            improvements=["Focal Loss"]
        )
        
        del model, optimizer, scheduler
        torch.cuda.empty_cache()
        
        return result
    
    def run_all_experiments(self, epochs_per_experiment=5):
        """运行所有消融实验"""
        print("\n" + "="*80)
        print("🎯 开始完整消融实验")
        print("="*80)
        print(f"\n配置:")
        print(f"   每个实验训练轮数: {epochs_per_experiment}")
        print(f"   预计总时间: ~{epochs_per_experiment * 10 * 10}分钟")
        print(f"   输出目录: {self.output_dir}")
        
        start_time = time.time()
        
        # 实验1: 基线
        self.run_baseline_experiment(epochs_per_experiment)
        
        # 实验2: + 注意力
        self.run_attention_experiment(epochs_per_experiment)
        
        # 实验3: + 特征工程
        self.run_feature_engineering_experiment(epochs_per_experiment)
        
        # 实验4: + Focal Loss
        self.run_focal_loss_experiment(epochs_per_experiment)
        
        # 更多实验可以继续添加...
        
        total_time = time.time() - start_time
        
        print("\n" + "="*80)
        print("🎉 所有实验完成!")
        print("="*80)
        print(f"总耗时: {total_time/60:.1f} 分钟")
        
        # 生成报告
        self.generate_report()
        
        return self.results
    
    def log_experiment(self, experiment_id, name, description, improvements):
        """记录实验信息"""
        self.experiment_log.append({
            'id': experiment_id,
            'name': name,
            'description': description,
            'improvements': improvements,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })
    
    def generate_report(self):
        """生成实验报告"""
        print("\n" + "="*80)
        print("📊 生成实验报告")
        print("="*80)
        
        # 保存结果为JSON
        report_path = os.path.join(self.output_dir, 'results.json')
        with open(report_path, 'w') as f:
            json.dump({
                'results': self.results,
                'experiments': self.experiment_log,
                'config': {
                    'seed': config.SEED,
                    'batch_size': config.BATCH_SIZE,
                    'learning_rate': config.LEARNING_RATE
                }
            }, f, indent=2)
        
        print(f"✅ JSON结果已保存: {report_path}")
        
        # 生成对比表格
        self._generate_comparison_table()
        
        # 生成可视化图表
        self._generate_visualizations()
        
        # 生成Markdown报告
        self._generate_markdown_report()
    
    def _generate_comparison_table(self):
        """生成对比表格"""
        df = pd.DataFrame(self.results)
        
        # 选择关键列
        comparison_df = df[[
            'model_name', 'best_val_auc', 'final_accuracy', 
            'final_f1', 'training_time'
        ]].copy()
        
        # 计算相对提升
        baseline_auc = comparison_df.iloc[0]['best_val_auc']
        comparison_df['auc_improvement'] = \
            (comparison_df['best_val_auc'] - baseline_auc) * 100
        
        # 保存为CSV
        csv_path = os.path.join(self.output_dir, 'comparison.csv')
        comparison_df.to_csv(csv_path, index=False)
        
        print(f"✅ 对比表格已保存: {csv_path}")
        
        # 打印表格
        print("\n" + "="*80)
        print("📈 实验结果对比")
        print("="*80)
        print(comparison_df.to_string(index=False))
    
    def _generate_visualizations(self):
        """生成可视化图表"""
        df = pd.DataFrame(self.results)
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. AUC对比
        ax1 = axes[0, 0]
        models = [r['model_name'] for r in self.results]
        aucs = [r['best_val_auc'] for r in self.results]
        colors = plt.cm.viridis(range(len(models)))
        
        bars = ax1.bar(range(len(models)), aucs, color=colors)
        ax1.set_xlabel('Model', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Best Validation AUC', fontsize=12, fontweight='bold')
        ax1.set_title('AUC Comparison Across Models', fontsize=14, fontweight='bold')
        ax1.set_xticks(range(len(models)))
        ax1.set_xticklabels([m.replace('_', '\n') for m in models], 
                           rotation=45, ha='right', fontsize=9)
        ax1.grid(axis='y', alpha=0.3)
        
        # 添加数值标签
        for i, (bar, auc) in enumerate(zip(bars, aucs)):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{auc:.4f}',
                    ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        # 2. 相对提升
        ax2 = axes[0, 1]
        baseline_auc = aucs[0]
        improvements = [(auc - baseline_auc) * 100 for auc in aucs]
        
        bars = ax2.bar(range(len(models)), improvements, color=colors)
        ax2.set_xlabel('Model', fontsize=12, fontweight='bold')
        ax2.set_ylabel('AUC Improvement (%)', fontsize=12, fontweight='bold')
        ax2.set_title('Relative Improvement over Baseline', fontsize=14, fontweight='bold')
        ax2.set_xticks(range(len(models)))
        ax2.set_xticklabels([m.replace('_', '\n') for m in models], 
                           rotation=45, ha='right', fontsize=9)
        ax2.axhline(y=0, color='r', linestyle='--', linewidth=2)
        ax2.grid(axis='y', alpha=0.3)
        
        for bar, imp in zip(bars, improvements):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{imp:+.2f}%',
                    ha='center', va='bottom' if imp >= 0 else 'top',
                    fontsize=10, fontweight='bold')
        
        # 3. 训练时间对比
        ax3 = axes[1, 0]
        times = [r['training_time'] / 60 for r in self.results]  # 转为分钟
        
        bars = ax3.bar(range(len(models)), times, color=colors)
        ax3.set_xlabel('Model', fontsize=12, fontweight='bold')
        ax3.set_ylabel('Training Time (minutes)', fontsize=12, fontweight='bold')
        ax3.set_title('Training Time Comparison', fontsize=14, fontweight='bold')
        ax3.set_xticks(range(len(models)))
        ax3.set_xticklabels([m.replace('_', '\n') for m in models], 
                           rotation=45, ha='right', fontsize=9)
        ax3.grid(axis='y', alpha=0.3)
        
        for bar, t in zip(bars, times):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height,
                    f'{t:.1f}m',
                    ha='center', va='bottom', fontsize=10)
        
        # 4. F1分数对比
        ax4 = axes[1, 1]
        f1_scores = [r['final_f1'] for r in self.results]
        
        bars = ax4.bar(range(len(models)), f1_scores, color=colors)
        ax4.set_xlabel('Model', fontsize=12, fontweight='bold')
        ax4.set_ylabel('F1 Score', fontsize=12, fontweight='bold')
        ax4.set_title('F1 Score Comparison', fontsize=14, fontweight='bold')
        ax4.set_xticks(range(len(models)))
        ax4.set_xticklabels([m.replace('_', '\n') for m in models], 
                           rotation=45, ha='right', fontsize=9)
        ax4.grid(axis='y', alpha=0.3)
        
        for bar, f1 in zip(bars, f1_scores):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height,
                    f'{f1:.4f}',
                    ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        
        # 保存图表
        viz_path = os.path.join(self.output_dir, 'ablation_comparison.png')
        plt.savefig(viz_path, dpi=300, bbox_inches='tight')
        print(f"✅ 可视化图表已保存: {viz_path}")
        
        plt.close()
    
    def _generate_markdown_report(self):
        """生成Markdown格式的报告"""
        report_lines = []
        
        # 标题
        report_lines.append("# 消融实验报告 (Ablation Study Report)")
        report_lines.append("")
        report_lines.append(f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append("")
        report_lines.append("---")
        report_lines.append("")
        
        # 实验配置
        report_lines.append("## 📋 实验配置")
        report_lines.append("")
        report_lines.append(f"- **随机种子**: {config.SEED}")
        report_lines.append(f"- **批次大小**: {config.BATCH_SIZE}")
        report_lines.append(f"- **学习率**: {config.LEARNING_RATE}")
        report_lines.append(f"- **数据规模**: 训练集 {len(self.train_df):,} / 验证集 {len(self.val_df):,}")
        report_lines.append("")
        report_lines.append("---")
        report_lines.append("")
        
        # 实验结果总表
        report_lines.append("## 📊 实验结果总览")
        report_lines.append("")
        
        # 创建表格
        report_lines.append("| 实验ID | 模型名称 | AUC | 准确率 | F1 | 训练时间 | 相对提升 |")
        report_lines.append("|--------|---------|-----|--------|----|---------|---------|")
        
        baseline_auc = self.results[0]['best_val_auc']
        
        for i, result in enumerate(self.results, 1):
            improvement = (result['best_val_auc'] - baseline_auc) * 100
            report_lines.append(
                f"| {i} | {result['model_name']} | "
                f"{result['best_val_auc']:.4f} | "
                f"{result['final_accuracy']:.4f} | "
                f"{result['final_f1']:.4f} | "
                f"{result['training_time']/60:.1f}min | "
                f"{improvement:+.2f}% |"
            )
        
        report_lines.append("")
        report_lines.append("---")
        report_lines.append("")
        
        # 关键发现
        report_lines.append("## 🔍 关键发现")
        report_lines.append("")
        
        # 找出最佳模型
        best_result = max(self.results, key=lambda x: x['best_val_auc'])
        report_lines.append(f"### 最佳模型")
        report_lines.append(f"- **名称**: {best_result['model_name']}")
        report_lines.append(f"- **AUC**: {best_result['best_val_auc']:.4f}")
        report_lines.append(f"- **相对基线提升**: {(best_result['best_val_auc'] - baseline_auc)*100:+.2f}%")
        report_lines.append("")
        
        # 各改进的贡献
        report_lines.append("### 各改进方向贡献")
        for i, result in enumerate(self.results):
            if i == 0:
                continue  # 跳过基线
            improvement = (result['best_val_auc'] - baseline_auc) * 100
            exp_info = self.experiment_log[i]
            report_lines.append(f"- **{exp_info['name']}**: {improvement:+.2f}%")
        
        report_lines.append("")
        report_lines.append("---")
        report_lines.append("")
        
        # 结论
        report_lines.append("## 💡 结论")
        report_lines.append("")
        report_lines.append(f"1. 基线模型AUC为 **{baseline_auc:.4f}**")
        report_lines.append(f"2. 最佳模型AUC达到 **{best_result['best_val_auc']:.4f}**")
        report_lines.append(f"3. 总体提升 **{(best_result['best_val_auc'] - baseline_auc)*100:.2f}%**")
        report_lines.append("")
        
        # 保存报告
        report_path = os.path.join(self.output_dir, 'REPORT.md')
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report_lines))
        
        print(f"✅ Markdown报告已保存: {report_path}")
        
        # 打印到控制台
        print("\n" + "="*80)
        print('\n'.join(report_lines))


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='消融实验')
    parser.add_argument('--epochs', type=int, default=5, 
                       help='每个实验的训练轮数 (默认: 5)')
    parser.add_argument('--output_dir', type=str, default='ablation_results',
                       help='输出目录 (默认: ablation_results)')
    
    args = parser.parse_args()
    
    print("\n" + "="*80)
    print("🎯 消融实验系统")
    print("="*80)
    print(f"\n配置:")
    print(f"   每个实验轮数: {args.epochs}")
    print(f"   输出目录: {args.output_dir}")
    print(f"   设备: {config.DEVICE}")
    
    # 创建实验管理器
    study = AblationStudy(output_dir=args.output_dir)
    
    # 运行所有实验
    results = study.run_all_experiments(epochs_per_experiment=args.epochs)
    
    print("\n" + "="*80)
    print("✅ 实验完成！请查看报告:")
    print(f"   - JSON: {args.output_dir}/results.json")
    print(f"   - CSV: {args.output_dir}/comparison.csv")
    print(f"   - 图表: {args.output_dir}/ablation_comparison.png")
    print(f"   - 报告: {args.output_dir}/REPORT.md")
    print("="*80)


if __name__ == '__main__':
    main()
