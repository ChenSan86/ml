"""
完整实验框架：基线模型 + 10个改进方向 + 消融实验
"""
import os
import time
import json
import torch
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from train import train as train_baseline
from config import config


class ExperimentRunner:
    """实验运行器"""
    
    def __init__(self, output_dir='experiments'):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.results = []
        self.start_time = datetime.now()
        
        # 实验配置
        self.experiments = {
            '0_baseline': {
                'name': '基线模型 (BiLSTM)',
                'script': 'train.py',
                'enabled': True,
                'priority': 0
            },
            '1_attention': {
                'name': '+ 注意力机制',
                'script': 'train_attention.py',
                'enabled': True,
                'priority': 1
            },
            '2_focal_loss': {
                'name': '+ Focal Loss',
                'script': 'train_focal.py',
                'enabled': True,
                'priority': 2
            },
            '3_feature_engineering': {
                'name': '+ 特征工程 (19个特征)',
                'script': 'train_enhanced.py',
                'enabled': True,
                'priority': 3
            },
            '4_data_augmentation': {
                'name': '+ 数据增强',
                'script': 'train_augmented.py',
                'enabled': True,
                'priority': 4
            },
            '5_contrastive': {
                'name': '+ 对比学习',
                'script': 'train_contrastive.py',
                'enabled': True,
                'priority': 5
            },
            '6_ensemble': {
                'name': '+ 模型集成 (K折)',
                'script': 'train_kfold.py --n_splits 5 --epochs 5',
                'enabled': True,
                'priority': 6
            },
            '7_label_smoothing': {
                'name': '+ 标签平滑',
                'script': 'train_label_smooth.py',
                'enabled': False,  # 需要实现
                'priority': 7
            },
            '8_pretrained_embeddings': {
                'name': '+ 预训练词向量',
                'script': 'train_pretrained.py',
                'enabled': False,  # 需要实现
                'priority': 8
            },
            '9_transformer': {
                'name': '+ Transformer架构',
                'script': 'train_transformer.py',
                'enabled': False,  # 需要实现
                'priority': 9
            },
            '10_all_combined': {
                'name': '全部组合 (最佳配置)',
                'script': 'train_full.py',
                'enabled': False,  # 需要实现
                'priority': 10
            }
        }
    
    def run_experiment(self, exp_id, exp_config):
        """运行单个实验"""
        print("\n" + "=" * 80)
        print(f"🔬 实验: {exp_config['name']}")
        print("=" * 80)
        
        exp_start = time.time()
        
        try:
            # 根据脚本名判断如何运行
            script = exp_config['script']
            
            if script == 'train.py':
                # 基线模型
                from train import train as train_fn
                model, history = train_fn()
                
            elif script == 'train_attention.py':
                # 注意力模型
                from train_attention import train as train_fn
                model, history = train_fn()
                
            elif script == 'train_enhanced.py':
                # 特征工程
                from train_enhanced import train_enhanced as train_fn
                model, history = train_fn()
                
            elif 'kfold' in script:
                # K折交叉验证
                os.system(f'python {script}')
                # 读取结果
                results_df = pd.read_csv('kfold_results.csv')
                best_auc = results_df['auc'].mean()
                history = {'best_auc': best_auc}
                
            else:
                # 其他脚本
                print(f"⚠️  脚本 {script} 尚未实现，跳过")
                return None
            
            # 提取结果
            if isinstance(history, dict) and 'val_auc' in history:
                best_auc = max(history['val_auc'])
                final_acc = history['val_metrics'][-1]['accuracy'] if 'val_metrics' in history else 0
                final_f1 = history['val_metrics'][-1]['f1'] if 'val_metrics' in history else 0
            elif isinstance(history, dict) and 'best_auc' in history:
                best_auc = history['best_auc']
                final_acc = 0
                final_f1 = 0
            else:
                best_auc = 0
                final_acc = 0
                final_f1 = 0
            
            exp_time = time.time() - exp_start
            
            result = {
                'exp_id': exp_id,
                'name': exp_config['name'],
                'auc': best_auc,
                'accuracy': final_acc,
                'f1': final_f1,
                'time_minutes': exp_time / 60,
                'status': 'success'
            }
            
            print(f"\n✅ 实验完成!")
            print(f"   AUC: {best_auc:.4f}")
            print(f"   用时: {exp_time/60:.1f} 分钟")
            
            return result
            
        except Exception as e:
            print(f"\n❌ 实验失败: {str(e)}")
            return {
                'exp_id': exp_id,
                'name': exp_config['name'],
                'auc': 0,
                'accuracy': 0,
                'f1': 0,
                'time_minutes': 0,
                'status': f'failed: {str(e)}'
            }
    
    def run_all_experiments(self, quick_mode=False):
        """运行所有实验"""
        print("\n" + "=" * 80)
        print("🚀 开始完整实验流程")
        print("=" * 80)
        print(f"开始时间: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"输出目录: {self.output_dir}")
        
        if quick_mode:
            print("⚡ 快速模式: 每个实验仅3个epoch")
            original_epochs = config.EPOCHS
            config.EPOCHS = 3
        
        # 按优先级运行实验
        sorted_exps = sorted(
            [(k, v) for k, v in self.experiments.items() if v['enabled']],
            key=lambda x: x[1]['priority']
        )
        
        for exp_id, exp_config in sorted_exps:
            result = self.run_experiment(exp_id, exp_config)
            if result:
                self.results.append(result)
                # 实时保存结果
                self.save_results()
        
        if quick_mode:
            config.EPOCHS = original_epochs
        
        # 生成报告
        self.generate_report()
        
        total_time = (datetime.now() - self.start_time).total_seconds() / 60
        print(f"\n🎉 全部实验完成! 总用时: {total_time:.1f} 分钟")
    
    def save_results(self):
        """保存结果到文件"""
        results_df = pd.DataFrame(self.results)
        results_df.to_csv(f'{self.output_dir}/results.csv', index=False)
        
        # 保存JSON格式
        with open(f'{self.output_dir}/results.json', 'w') as f:
            json.dump(self.results, f, indent=2)
    
    def generate_report(self):
        """生成实验报告"""
        print("\n" + "=" * 80)
        print("📊 生成实验报告")
        print("=" * 80)
        
        if not self.results:
            print("⚠️  没有实验结果")
            return
        
        df = pd.DataFrame(self.results)
        
        # 1. 控制台报告
        self._print_console_report(df)
        
        # 2. 可视化报告
        self._generate_visualizations(df)
        
        # 3. Markdown报告
        self._generate_markdown_report(df)
        
        print(f"\n✅ 报告已生成: {self.output_dir}/")
    
    def _print_console_report(self, df):
        """打印控制台报告"""
        print("\n" + "=" * 80)
        print("📈 实验结果汇总")
        print("=" * 80)
        
        print(f"\n{'实验名称':<40} {'AUC':>8} {'提升':>8} {'用时(分)':>10}")
        print("-" * 80)
        
        baseline_auc = df.iloc[0]['auc'] if len(df) > 0 else 0
        
        for idx, row in df.iterrows():
            improvement = (row['auc'] - baseline_auc) * 100 if baseline_auc > 0 else 0
            improvement_str = f"+{improvement:.2f}%" if improvement > 0 else f"{improvement:.2f}%"
            
            print(f"{row['name']:<40} {row['auc']:>8.4f} {improvement_str:>8} {row['time_minutes']:>10.1f}")
        
        print("-" * 80)
        print(f"\n🏆 最佳模型: {df.loc[df['auc'].idxmax(), 'name']}")
        print(f"   AUC: {df['auc'].max():.4f}")
        print(f"   相比基线提升: +{(df['auc'].max() - baseline_auc) * 100:.2f}%")
    
    def _generate_visualizations(self, df):
        """生成可视化图表"""
        # 1. AUC对比图
        plt.figure(figsize=(14, 8))
        
        colors = ['#FF6B6B' if i == 0 else '#4ECDC4' if i == len(df)-1 else '#95E1D3' 
                  for i in range(len(df))]
        
        bars = plt.barh(df['name'], df['auc'], color=colors)
        
        # 添加数值标签
        for i, (bar, auc) in enumerate(zip(bars, df['auc'])):
            plt.text(auc + 0.001, bar.get_y() + bar.get_height()/2, 
                    f'{auc:.4f}', va='center', fontsize=10, fontweight='bold')
        
        plt.xlabel('AUC Score', fontsize=12, fontweight='bold')
        plt.title('实验结果对比', fontsize=16, fontweight='bold', pad=20)
        plt.xlim(0.94, df['auc'].max() * 1.01)
        plt.grid(axis='x', alpha=0.3, linestyle='--')
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/auc_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. 提升曲线图
        plt.figure(figsize=(12, 6))
        baseline_auc = df.iloc[0]['auc']
        improvements = [(auc - baseline_auc) * 100 for auc in df['auc']]
        
        plt.plot(range(len(df)), improvements, marker='o', linewidth=2, 
                markersize=10, color='#FF6B6B')
        plt.fill_between(range(len(df)), 0, improvements, alpha=0.3, color='#FF6B6B')
        
        plt.xticks(range(len(df)), df['name'], rotation=45, ha='right')
        plt.ylabel('AUC提升 (%)', fontsize=12, fontweight='bold')
        plt.title('累积改进效果', fontsize=16, fontweight='bold', pad=20)
        plt.axhline(y=0, color='gray', linestyle='--', linewidth=1)
        plt.grid(alpha=0.3, linestyle='--')
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/improvement_curve.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. 时间开销对比
        plt.figure(figsize=(12, 6))
        plt.bar(range(len(df)), df['time_minutes'], color='#95E1D3', edgecolor='black')
        plt.xticks(range(len(df)), df['name'], rotation=45, ha='right')
        plt.ylabel('训练时间 (分钟)', fontsize=12, fontweight='bold')
        plt.title('训练时间对比', fontsize=16, fontweight='bold', pad=20)
        plt.grid(axis='y', alpha=0.3, linestyle='--')
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/time_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ 可视化图表已保存")
    
    def _generate_markdown_report(self, df):
        """生成Markdown格式报告"""
        baseline_auc = df.iloc[0]['auc']
        
        report = f"""# 文本语义匹配模型 - 完整实验报告

## 📊 实验概述

**实验日期**: {self.start_time.strftime('%Y年%m月%d日')}  
**总实验数**: {len(df)}  
**总用时**: {df['time_minutes'].sum():.1f} 分钟  

---

## 🎯 实验结果汇总

### 性能对比表

| 实验ID | 实验名称 | AUC | 准确率 | F1 | 提升 | 用时(分) |
|--------|---------|-----|--------|----|----|---------|
"""
        
        for idx, row in df.iterrows():
            improvement = (row['auc'] - baseline_auc) * 100
            improvement_str = f"+{improvement:.2f}%" if improvement >= 0 else f"{improvement:.2f}%"
            
            report += f"| {idx+1} | {row['name']} | {row['auc']:.4f} | "
            report += f"{row['accuracy']:.4f} | {row['f1']:.4f} | "
            report += f"{improvement_str} | {row['time_minutes']:.1f} |\n"
        
        report += f"""
---

## 🏆 最佳模型

**模型**: {df.loc[df['auc'].idxmax(), 'name']}  
**AUC**: {df['auc'].max():.4f}  
**相比基线提升**: +{(df['auc'].max() - baseline_auc) * 100:.2f}%  
**训练时间**: {df.loc[df['auc'].idxmax(), 'time_minutes']:.1f} 分钟

---

## 📈 关键发现

### 1. 各改进方向效果

"""
        
        for idx, row in df.iterrows():
            if idx == 0:
                report += f"- **基线模型**: AUC {row['auc']:.4f} (参考基准)\n"
            else:
                improvement = (row['auc'] - df.iloc[idx-1]['auc']) * 100
                report += f"- **{row['name']}**: "
                if improvement > 0:
                    report += f"✅ +{improvement:.2f}% (AUC {row['auc']:.4f})\n"
                elif improvement < 0:
                    report += f"⚠️  {improvement:.2f}% (AUC {row['auc']:.4f})\n"
                else:
                    report += f"➡️  持平 (AUC {row['auc']:.4f})\n"
        
        report += f"""
### 2. 性能vs成本分析

"""
        df['improvement'] = (df['auc'] - baseline_auc) * 100
        df['efficiency'] = df['improvement'] / df['time_minutes']
        
        best_efficiency = df.loc[df['efficiency'].idxmax()]
        report += f"**最高效改进**: {best_efficiency['name']}\n"
        report += f"- 提升: +{best_efficiency['improvement']:.2f}%\n"
        report += f"- 用时: {best_efficiency['time_minutes']:.1f}分钟\n"
        report += f"- 效率: {best_efficiency['efficiency']:.3f}% per minute\n\n"
        
        report += """
---

## 💡 消融实验结论

### 有效的改进方向
"""
        effective = df[df['improvement'] > 0.5].iloc[1:]  # 排除基线
        if len(effective) > 0:
            for _, row in effective.iterrows():
                report += f"- ✅ {row['name']}: +{row['improvement']:.2f}%\n"
        else:
            report += "（无显著提升的改进）\n"
        
        report += """
### 改进较小的方向
"""
        minor = df[(df['improvement'] >= 0) & (df['improvement'] <= 0.5)].iloc[1:]
        if len(minor) > 0:
            for _, row in minor.iterrows():
                report += f"- ⚠️  {row['name']}: +{row['improvement']:.2f}%\n"
        else:
            report += "（无）\n"
        
        report += """
---

## 🎨 可视化结果

### AUC对比图
![AUC对比](auc_comparison.png)

### 改进曲线
![改进曲线](improvement_curve.png)

### 时间开销
![时间对比](time_comparison.png)

---

## 🔍 详细分析

### 基线模型性能
"""
        baseline = df.iloc[0]
        report += f"""
- **架构**: BiLSTM双塔 + 交互层
- **参数量**: ~1350万
- **AUC**: {baseline['auc']:.4f}
- **准确率**: {baseline['accuracy']:.4f}
- **F1分数**: {baseline['f1']:.4f}

### 最优配置建议

根据实验结果，推荐以下配置：
"""
        
        # 找出提升最大的前3个
        top3 = df.nlargest(3, 'auc').iloc[1:]  # 排除可能的基线
        for idx, (_, row) in enumerate(top3.iterrows(), 1):
            report += f"{idx}. {row['name']}\n"
        
        report += f"""
**预期性能**: AUC {df['auc'].max():.4f}

---

## 📝 实验配置

- **数据集**: 40万训练样本
- **验证集**: 10% (4万样本)
- **Batch Size**: 256
- **初始学习率**: 0.001
- **优化器**: AdamW
- **训练轮数**: 10 epochs (每个实验)

---

**报告生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
        
        # 保存报告
        with open(f'{self.output_dir}/EXPERIMENT_REPORT.md', 'w', encoding='utf-8') as f:
            f.write(report)
        
        print(f"✅ Markdown报告已保存: {self.output_dir}/EXPERIMENT_REPORT.md")


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='运行完整实验')
    parser.add_argument('--quick', action='store_true', help='快速模式(3 epochs)')
    parser.add_argument('--output', type=str, default='experiments', help='输出目录')
    
    args = parser.parse_args()
    
    # 创建实验运行器
    runner = ExperimentRunner(output_dir=args.output)
    
    # 运行所有实验
    runner.run_all_experiments(quick_mode=args.quick)


if __name__ == '__main__':
    main()
