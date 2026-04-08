#!/usr/bin/env bash
set -euo pipefail

# 测试 param_iter 运行结果的 si 指标

# 数据集
DATASETS=(
  com-lj
)

# 方法列表
METHODS=(
  "snap_tfdp"
  "snap_tfdp_par"
  "snap_tfdp_gpu"
)

# 迭代次数
ITERS=(
  1
  2
  5
  10
  20
  30
  40
  50
  60
  70
  80
  90
  100
)

# 种子数
SEEDS=(1 2 3 4 5)

# 指标列表
METRICS=(
  "si"
)

# 参数配置
DATA_DIR="./data"
RESULT_BASE_DIR="./results/param-iter"
METRICS_OUTPUT_DIR="./statistics/param-iter/metrics"

# 创建输出目录
mkdir -p "${METRICS_OUTPUT_DIR}"

echo "===== Running metrics for param_iter ====="
echo "Methods: ${METHODS[*]}"
echo "Datasets: ${DATASETS[*]}"
echo "Iterations: ${ITERS[*]}"
echo "Seeds: ${SEEDS[*]}"
echo "Metrics: ${METRICS[*]}"
echo ""

# 对每个指标和方法运行指标计算和合并
for METRIC in "${METRICS[@]}"; do
  echo ""
  echo "===== Processing metric: ${METRIC} ====="

  for METHOD in "${METHODS[@]}"; do
  echo ""
  echo "===== Processing method: ${METHOD} ====="

  # 构建方法列表：{method}_iter_{iter}_seed_{seed}
  METHODS_LIST=()
  for iter in "${ITERS[@]}"; do
    for seed in "${SEEDS[@]}"; do
      METHODS_LIST+=("${METHOD}_iter_${iter}_seed_${seed}")
    done
  done

  # 构建 methods 参数字符串
  METHODS_STR=$(IFS=,; echo "${METHODS_LIST[*]}")

  # 构建 datasets 参数字符串
  DATASETS_STR=$(IFS=,; echo "${DATASETS[*]}")

  # 构建 metrics 参数字符串
  METRICS_STR="${METRIC}"

  echo "Methods count: ${#METHODS_LIST[@]}"
  echo "Running metrics calculation..."
  echo ""

  python tools/metrics_all.py \
    --datasets "${DATASETS_STR}" \
    --methods "${METHODS_STR}" \
    --metrics "${METRICS_STR}" \
    --data-dir "${DATA_DIR}" \
    --result-base-dir "${RESULT_BASE_DIR}/${METHOD}" \
    --metrics-output-dir "${METRICS_OUTPUT_DIR}"

  echo ""
  echo "===== Merging CSV files for ${METHOD} ====="

  # 合并 CSV 文件：将 dataset.method_iter_X_seed_Y.{metric}.csv 合并为 dataset.method.{metric}.csv
  for dataset in "${DATASETS[@]}"; do
    MERGED="${METRICS_OUTPUT_DIR}/${dataset}.${METHOD}.${METRIC}.csv"
    TEMP_MERGED="${METRICS_OUTPUT_DIR}/.temp.${dataset}.${METHOD}.${METRIC}.csv"

    echo "Merging ${dataset} results..."
    first=true
    found_files=false
    seed_files=()

    for iter in "${ITERS[@]}"; do
      for seed in "${SEEDS[@]}"; do
        seed_file="${METRICS_OUTPUT_DIR}/${dataset}.${METHOD}_iter_${iter}_seed_${seed}.${METRIC}.csv"
        if [[ -f "${seed_file}" ]]; then
          found_files=true
          seed_files+=("${seed_file}")
          if [[ "${first}" == true ]]; then
            cat "${seed_file}" >> "${TEMP_MERGED}"
            first=false
          else
            tail -n +2 "${seed_file}" >> "${TEMP_MERGED}"
          fi
        fi
      done
    done

    # 只有找到 seed 文件才覆盖原文件
    if [[ "${found_files}" == true ]]; then
      mv "${TEMP_MERGED}" "${MERGED}"
      # 删除已合并的分 seed 文件
      for seed_file in "${seed_files[@]}"; do
        rm -f "${seed_file}"
      done
      echo "✓ ${dataset} merged, deleted ${#seed_files[@]} seed files"
    else
      rm -f "${TEMP_MERGED}"
      echo "⚠ ${dataset} no seed files found, skipped"
    fi
  done

  echo "✓ ${METHOD} completed"
done

  echo "✓ ${METRIC} completed"
done