#!/usr/bin/env bash
set -euo pipefail

# 数据集
DATASETS=(
  com-lj
)

# 方法列表
METHODS=(
  "nsgl"
  "nsgl_par"
  "nsgl_gpu_8192"
)

# 迭代值列表
ITERS=(
  5 10 20 30 40 50 60 70 80 90 100
)

# 参数配置
BASE_DIR="./statistics/param-iter/metrics"
OUTPUT_DIR="./figures/param-iter"

# 构建 datasets 参数
DATASETS_STR=$(IFS=' '; echo "${DATASETS[*]}")

# 构建 methods 参数
METHODS_STR=$(IFS=' '; echo "${METHODS[*]}")

# 构建 iters 参数
ITERS_STR=$(IFS=' '; echo "${ITERS[*]}")

echo "===== Drawing parameter iteration metric curves ====="
echo "Datasets: ${DATASETS_STR}"
echo "Methods: ${METHODS_STR}"
echo "Iterations: ${ITERS_STR}"
echo "Base directory: ${BASE_DIR}"
echo "Output directory: ${OUTPUT_DIR}"
echo ""

python3 scripts/param-iter/plot_param_iter.py \
  --methods ${METHODS_STR} \
  --base-dir "${BASE_DIR}" \
  --output-dir "${OUTPUT_DIR}" \
  --datasets ${DATASETS_STR} \
  --iters ${ITERS_STR}

echo ""
echo "===== Metric curve drawing completed ====="