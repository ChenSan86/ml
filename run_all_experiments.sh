#!/bin/bash
################################################################################
# 完整实验脚本：基线 + 10个改进方向 + 消融实验 + 报告生成
################################################################################

echo "================================================================================"
echo "🚀 文本语义匹配模型 - 完整实验流程"
echo "================================================================================"
echo "开始时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo ""

# 设置输出目录
OUTPUT_DIR="experiments_$(date '+%Y%m%d_%H%M%S')"
mkdir -p $OUTPUT_DIR
cd /home/xinguanze/class/ml2

# 运行模式: full | quick
MODE=${1:-quick}

if [ "$MODE" == "quick" ]; then
    echo "⚡ 快速模式: 每个实验3个epoch"
    EPOCHS=3
    python run_experiments.py --quick --output $OUTPUT_DIR
else
    echo "🔬 完整模式: 每个实验10个epoch"
    EPOCHS=10
    python run_experiments.py --output $OUTPUT_DIR
fi

echo ""
echo "================================================================================"
echo "✅ 实验完成!"
echo "================================================================================"
echo "结束时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo "结果目录: $OUTPUT_DIR/"
echo ""
echo "查看报告:"
echo "  - 详细报告: cat $OUTPUT_DIR/EXPERIMENT_REPORT.md"
echo "  - CSV结果:  cat $OUTPUT_DIR/results.csv"
echo "  - 图表: ls $OUTPUT_DIR/*.png"
echo ""
