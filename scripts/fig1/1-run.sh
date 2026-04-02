#!/usr/bin/env bash
set -euo pipefail

# Fig1: NSGL parallel method with different epochs
# Dataset: com-orkut-select
# Method: NSGL parallel (16 threads)
# Epochs: 0, 1, 5, 30

DATA_DIR="./data"
DATASET="com-orkut-select"
RESULT_DIR="./results/fig1"
EPOCHS=(0 1 5 30)

# 创建结果目录
mkdir -p "${RESULT_DIR}"

echo "===== Running Fig1 experiments ====="
echo "Dataset: ${DATASET}"
echo "Method: Parallel (16 threads)"
echo "Epochs: ${EPOCHS[*]}"
echo "Result directory: ${RESULT_DIR}"
echo ""

for epoch in "${EPOCHS[@]}"; do
  echo "===== Running epoch: ${epoch} ====="

  LOG_FILE="${RESULT_DIR}/${DATASET}.epoch_${epoch}.log"

  echo "Start time: $(date)" | tee "${LOG_FILE}"

  /usr/bin/time -v bash -c "
    ./snap-tfdp \
      '${DATA_DIR}/${DATASET}.txt' \
      '${RESULT_DIR}/${DATASET}.txt' \
      --init pmds \
      --pmds-file 'data/PMDS_init/${DATASET}.txt' \
      --k 5 \
      --n-epoch ${epoch} \
      --parallel \
      --n-threads 16 \
  " 2>&1 | tee -a "${LOG_FILE}"

  echo "End time: $(date)" | tee -a "${LOG_FILE}"
  echo "" | tee -a "${LOG_FILE}"

  # 重命名布局结果文件
  LAYOUT_FILE="${RESULT_DIR}/${DATASET}.txt"
  LAYOUT_RENAMED="${RESULT_DIR}/${DATASET}.epoch_${epoch}.layout.txt"
  if [[ -f "${LAYOUT_FILE}" ]]; then
    mv "${LAYOUT_FILE}" "${LAYOUT_RENAMED}"
    echo "✓ Saved layout: ${LAYOUT_RENAMED}"
  fi
done

echo ""
echo "===== Fig1 experiments completed ====="
