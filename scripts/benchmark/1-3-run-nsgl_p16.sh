#!/usr/bin/env bash
set -euo pipefail

DATA_DIR="./data"
SEED="${1:-}"
if [ -n "$SEED" ]; then
  RESULT_DIR="./results/benchmark/__seed${SEED}/snap_tfdp_par"
  PMDS_DIR="./results/benchmark/__seed${SEED}/pmds"
else
  RESULT_DIR="./results/benchmark/snap_tfdp_par"
  PMDS_DIR="./results/benchmark/pmds"
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