#!/usr/bin/env bash
set -euo pipefail

DATA_DIR="./data"
SEED="${1:-}"
if [ -n "$SEED" ]; then
  PMDS_DIR="./results/benchmark/__seed${SEED}/pmds"
  RESULT_DIR="./results/benchmark/__seed${SEED}/fa2"
else
  PMDS_DIR="./results/benchmark/pmds"
  RESULT_DIR="./results/benchmark/fa2"
fi

DATASETS=(
  APH
#  aircraft
  co_author_8391
  socfb-Yale4
  ACO
#  socfb-UF21
#  soc-Flickr-ASU
#  com-dblp
#  com-amazon

#  com-youtube
#  com-orkut
#  com-lj
)

mkdir -p "${RESULT_DIR}"

for dataset in "${DATASETS[@]}"; do
  LOG_FILE="${RESULT_DIR}/${dataset}.log"
  OUTPUT_FILE="${RESULT_DIR}/${dataset}.txt"

  echo "===== Running dataset: ${dataset} =====" | tee "${LOG_FILE}"
  echo "Start time: $(date)" | tee -a "${LOG_FILE}"

  /usr/bin/time -v bash -c "
    python third_party/FA2/run_fa2.py \
      '${DATA_DIR}/${dataset}.mtx' \
      '${PMDS_DIR}/${dataset}.txt' \
      '${OUTPUT_FILE}'
  " 2>&1 | tee -a "${LOG_FILE}"

  echo "End time: $(date)" | tee -a "${LOG_FILE}"
  echo "" | tee -a "${LOG_FILE}"

done