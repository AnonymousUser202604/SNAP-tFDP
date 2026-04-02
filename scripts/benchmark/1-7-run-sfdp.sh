#!/usr/bin/env bash
set -euo pipefail

DATA_DIR="./data"
SEED="${1:-}"
if [ -n "$SEED" ]; then
  RESULT_DIR="./results/benchmark/__seed${SEED}/sfdp"
  SFDP_SEED="${SEED}"
else
  RESULT_DIR="./results/benchmark/sfdp"
  SFDP_SEED="42"
fi
POSTPROCESS="python ./third_party/sfdp/postprocess.py"

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
  DOT_FILE="${DATA_DIR}/${dataset}.dot"
  TEMP_FILE="${RESULT_DIR}/${dataset}_temp.txt"
  OUTPUT_FILE="${RESULT_DIR}/${dataset}.txt"

  # 检查输入文件
  if [ ! -f "${DOT_FILE}" ]; then
    echo "===== Skipping ${dataset}: dot file not found =====" | tee "${LOG_FILE}"
    continue
  fi

  echo "===== Running dataset: ${dataset} =====" | tee "${LOG_FILE}"
  echo "Start time: $(date)" | tee -a "${LOG_FILE}"

  /usr/bin/time -v bash -c "
    sfdp '${DOT_FILE}' -Gstart=random -Gseed=${SFDP_SEED} -Tplain | grep '^node' | awk '{print \$2,\$3,\$4}' > '${TEMP_FILE}' && \
    ${POSTPROCESS} '${TEMP_FILE}' '${OUTPUT_FILE}' && \
    rm '${TEMP_FILE}'
  " 2>&1 | tee -a "${LOG_FILE}"

  echo "End time: $(date)" | tee -a "${LOG_FILE}"
  echo "" | tee -a "${LOG_FILE}"

done