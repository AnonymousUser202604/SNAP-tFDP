#!/usr/bin/env bash
set -euo pipefail

DATA_DIR="./data"
SEED="${1:-}"
if [ -n "$SEED" ]; then
  RESULT_DIR="./results/benchmark/__seed${SEED}/linlog"
  PMDS_DIR="./results/benchmark/__seed${SEED}/pmds"
else
  RESULT_DIR="./results/benchmark/linlog"
  PMDS_DIR="./results/benchmark/pmds"
fi
LINLOG_DIR="./third_party/LinLogLayout"
POSTPROCESS="${LINLOG_DIR}/postprocess.py"

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

# 编译 Java 源码
echo "===== Compiling LinLogLayout ====="
javac -d "${LINLOG_DIR}/bin" "${LINLOG_DIR}/src"/*.java
echo "Compile done."
echo ""

for dataset in "${DATASETS[@]}"; do
  LOG_FILE="${RESULT_DIR}/${dataset}.log"
  RAW_FILE="${RESULT_DIR}/${dataset}_raw.txt"
  OUTPUT_FILE="${RESULT_DIR}/${dataset}.txt"

  echo "===== Running dataset: ${dataset} =====" | tee "${LOG_FILE}"
  echo "Start time: $(date)" | tee -a "${LOG_FILE}"

  # 布局
  /usr/bin/time -v bash -c "
    java -cp '${LINLOG_DIR}/bin' LinLogLayout 2 \
      '${DATA_DIR}/${dataset}.mtx' \
      '${RAW_FILE}'
  " 2>&1 | tee -a "${LOG_FILE}"

  # 后处理
  python "${POSTPROCESS}" "${RAW_FILE}" "${OUTPUT_FILE}" 2>&1 | tee -a "${LOG_FILE}"

  echo "End time: $(date)" | tee -a "${LOG_FILE}"
  echo "" | tee -a "${LOG_FILE}"

done