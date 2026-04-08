#!/usr/bin/env bash
set -euo pipefail

DATA_DIR="./data"
PMDS_DIR="./data/PMDS_init"
RESULT_BASE_DIR="./results/param-k"

DATASETS=(
  APH
  aircraft
  co_author_8391
  socfb-Yale4
  ACO
  socfb-UF21
  soc-Flickr-ASU
  com-dblp
  com-amazon
  com-youtube
  com-orkut
  com-lj
)

if [ $# -gt 0 ]; then
  K_VALUES=("$@")
else
  K_VALUES=(1 2 3 4 5 6 7 8 9 10)
fi

mkdir -p "${RESULT_BASE_DIR}"

for k in "${K_VALUES[@]}"; do
  K_DIR="${RESULT_BASE_DIR}/k_${k}"
  mkdir -p "${K_DIR}"

  for dataset in "${DATASETS[@]}"; do
    OUTPUT_FILE="${K_DIR}/${dataset}.txt"
    LOG_FILE="${K_DIR}/${dataset}.log"

    echo "===== Running dataset: ${dataset}, k=${k} =====" | tee "${LOG_FILE}"
    echo "Start time: $(date)" | tee -a "${LOG_FILE}"

    /usr/bin/time -v bash -c "
      ./snap-tfdp \
        '${DATA_DIR}/${dataset}.txt' \
        '${K_DIR}/${dataset}.txt' \
        --init pmds \
        --pmds-file '${PMDS_DIR}/${dataset}.txt' \
        --k ${k} \
        --n-epoch 50 \
    " 2>&1 | tee -a "${LOG_FILE}"

    echo "End time: $(date)" | tee -a "${LOG_FILE}"
    echo "" | tee -a "${LOG_FILE}"

  done
done