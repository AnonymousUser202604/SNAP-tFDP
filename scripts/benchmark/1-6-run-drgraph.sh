#!/usr/bin/env bash
set -euo pipefail

DATASET_DIR="./data"
SEED="${1:-}"
if [ -n "$SEED" ]; then
  RESULT_DIR="./results/benchmark/__seed${SEED}/drgraph"
else
  RESULT_DIR="./results/benchmark/drgraph"
fi

DATASETS=(
  APH
  aircraft
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
    ./third_party/DRGraph/Vis \
      -input '${DATASET_DIR}/${dataset}.txt' \
      -output '${OUTPUT_FILE}' \
      -neg 5 \
      -samples 400 \
      -gamma 0.1 \
      -mode 1 \
      -A 2 \
      -B 1 \
      -threads 1
  " 2>&1 | tee -a "${LOG_FILE}"

  echo "End time: $(date)" | tee -a "${LOG_FILE}"
  echo "" | tee -a "${LOG_FILE}"

  if [[ -f "${OUTPUT_FILE}" ]]; then
    sed -i '1d' "${OUTPUT_FILE}"
  fi

done