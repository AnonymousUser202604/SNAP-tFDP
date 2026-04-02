#!/usr/bin/env bash
set -euo pipefail

DATA_DIR="./data"
RESULT_DIR="./results/biggest/nsgl_par"
PMDS_DIR="./results/benchmark/pmds"

DATASETS=(
  com-friendster
)

mkdir -p "${RESULT_DIR}"

for dataset in "${DATASETS[@]}"; do
  LOG_FILE="${RESULT_DIR}/${dataset}.log"

  echo "===== Running dataset: ${dataset} =====" | tee "${LOG_FILE}"
  echo "Start time: $(date)" | tee -a "${LOG_FILE}"

  /usr/bin/time -v bash -c "
    ./snap-tfdp \
      '${DATA_DIR}/${dataset}.txt' \
      '${RESULT_DIR}/${dataset}.txt' \
      --init pmds \
      --pmds-file '${PMDS_DIR}/${dataset}.txt' \
      --k 3 \
      --n-epoch 50 \
      --parallel \
      --n-threads 16 \
  " 2> >(tee -a "${LOG_FILE}" >&2)

  echo "End time: $(date)" | tee -a "${LOG_FILE}"
  echo "" | tee -a "${LOG_FILE}"

done