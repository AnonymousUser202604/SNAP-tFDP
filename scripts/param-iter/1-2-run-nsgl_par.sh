#!/usr/bin/env bash
set -euo pipefail

DATA_DIR="./data"
RESULT_DIR="./results/param-iter/snap_tfdp_par"

DATASETS=(
  com-lj
)

ITERS=(
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

mkdir -p "${RESULT_DIR}"

for dataset in "${DATASETS[@]}"; do
  for iter in "${ITERS[@]}"; do
    for seed in {1..5}; do
      # 生成随机种子
      random_seed=$((RANDOM * 32768 + RANDOM))

      LOG_FILE="${RESULT_DIR}/${dataset}.iter_${iter}.seed_${seed}.log"

      echo "===== Running dataset: ${dataset}, iter: ${iter}, seed: ${seed} (Parallel) =====" | tee "${LOG_FILE}"
      echo "Start time: $(date)" | tee -a "${LOG_FILE}"

      /usr/bin/time -v bash -c "
        ./snap-tfdp \
          '${DATA_DIR}/${dataset}.txt' \
          '${RESULT_DIR}/${dataset}.txt' \
          --seed ${random_seed} \
          --neg 3 \
          --n-epoch ${iter} \
          --parallel \
          --n-threads 16 \
      " 2>&1 | tee -a "${LOG_FILE}"

      echo "End time: $(date)" | tee -a "${LOG_FILE}"
      echo "" | tee -a "${LOG_FILE}"

      # 重命名布局结果文件
      LAYOUT_FILE="${RESULT_DIR}/${dataset}.txt"
      LAYOUT_RENAMED="${RESULT_DIR}/${dataset}.iter_${iter}.seed_${seed}.layout.txt"
      if [[ -f "${LAYOUT_FILE}" ]]; then
        mv "${LAYOUT_FILE}" "${LAYOUT_RENAMED}"
      fi

    done
  done
done
