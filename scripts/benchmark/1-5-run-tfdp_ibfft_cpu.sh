#!/usr/bin/env bash
set -euo pipefail

ALGO="ibFFT_CPU"
DATA_DIR="./data"
SEED="${1:-}"
if [ -n "$SEED" ]; then
  RESULT_DIR="./results/benchmark/__seed${SEED}/tfdp_ibfft_cpu"
  PMDS_DIR="./results/benchmark/__seed${SEED}/pmds"
else
  RESULT_DIR="./results/benchmark/tfdp_ibfft_cpu"
  PMDS_DIR="./results/benchmark/pmds"
fi
RUN_TFDP="python ./third_party/t-fdp/run_tfdp.py"

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
    ${RUN_TFDP} \
      '${DATA_DIR}/${dataset}.mtx' \
      '${PMDS_DIR}/${dataset}.txt' \
      '${OUTPUT_FILE}' \
      --algo '${ALGO}'
  " 2>&1 | tee -a "${LOG_FILE}"

  echo "End time: $(date)" | tee -a "${LOG_FILE}"
  echo "" | tee -a "${LOG_FILE}"

done