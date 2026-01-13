#!/bin/bash
# 消融实验运行脚本

echo "=================================="
echo "🎯 消融实验系统"
echo "=================================="
echo ""

# 设置参数
EPOCHS=${1:-5}  # 默认每个实验5个epoch
OUTPUT_DIR=${2:-ablation_results}

echo "配置:"
echo "  - 每个实验轮数: $EPOCHS"
echo "  - 输出目录: $OUTPUT_DIR"
echo ""

# 运行消融实验
python ablation_study.py \
    --epochs $EPOCHS \
    --output_dir $OUTPUT_DIR

echo ""
echo "=================================="
echo "✅ 消融实验完成!"
echo "=================================="
echo ""
echo "查看结果:"
echo "  - 完整报告: cat $OUTPUT_DIR/REPORT.md"
echo "  - 对比表格: cat $OUTPUT_DIR/comparison.csv"
echo "  - 可视化图: open $OUTPUT_DIR/ablation_comparison.png"
echo ""
