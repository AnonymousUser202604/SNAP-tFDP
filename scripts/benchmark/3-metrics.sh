#!/usr/bin/env bash
set -euo pipefail

# 所有方法列表
METHODS=(
  "fr"
  "pmds"
  "sfdp"
  "linlog"
  "fa2"
  "tsnet"
  "drgraph"
  "tfdp"
  "nsgl"
  "nsgl_par"
  "nsgl_gpu"
)

# 所有数据集列表
DATASETS=(
  "APH"
  "aircraft"
  "co_author_8391"
  "socfb-Yale4"
  "ACO"
  "socfb-UF21"
  "soc-Flickr-ASU"
  "com-dblp"
  "com-amazon"
  "com-youtube"
  "com-orkut"
  "com-lj"
)

# 指标列表
METRICS=(
  "ari"
  "si"
  "np2"
)

# 参数配置
DATA_DIR="./data"
RESULT_ROOT_DIR="./results/benchmark"
METRICS_ROOT_DIR="./statistics/benchmark"

mkdir -p "${METRICS_ROOT_DIR}"

# 构建 methods/datasets/metrics 参数
METHODS_STR=$(IFS=,; echo "${METHODS[*]}")
DATASETS_STR=$(IFS=,; echo "${DATASETS[*]}")
METRICS_STR=$(IFS=,; echo "${METRICS[*]}")

# 扫描所有 seed 结果目录
SEED_RESULT_DIRS=()
for seed_result_dir in "${RESULT_ROOT_DIR}"/__seed*; do
  if [[ -d "${seed_result_dir}" ]]; then
    SEED_RESULT_DIRS+=("${seed_result_dir}")
  fi
done

if [[ ${#SEED_RESULT_DIRS[@]} -eq 0 ]]; then
  echo "No seed result directories found under ${RESULT_ROOT_DIR}/__seed*"
  exit 1
fi

echo "===== Running metrics per seed ====="
echo "Methods: ${METHODS_STR}"
echo "Datasets: ${DATASETS_STR}"
echo "Metrics: ${METRICS_STR}"
echo "Data directory: ${DATA_DIR}"
echo ""

SEED_NAMES=()
for seed_result_dir in "${SEED_RESULT_DIRS[@]}"; do
  seed_name=$(basename "${seed_result_dir}")
  seed_metrics_dir="${METRICS_ROOT_DIR}/${seed_name}"

  SEED_NAMES+=("${seed_name}")
  mkdir -p "${seed_metrics_dir}"

  echo "===== Seed: ${seed_name} ====="
  echo "Result base directory: ${seed_result_dir}"
  echo "Metrics output directory: ${seed_metrics_dir}"

  python tools/metrics_all.py \
    --methods "${METHODS_STR}" \
    --datasets "${DATASETS_STR}" \
    --metrics "${METRICS_STR}" \
    --data-dir "${DATA_DIR}" \
    --result-base-dir "${seed_result_dir}" \
    --metrics-output-dir "${seed_metrics_dir}"

  echo ""
done

echo "===== Per-seed metrics calculation completed ====="

echo ""
echo "===== Merging CSV files by metric (average across seeds) ====="

# 判断是否为数值
is_number() {
  local value="$1"
  [[ "${value}" =~ ^-?[0-9]+([.][0-9]+)?([eE]-?[0-9]+)?$ ]]
}

# 对每个指标进行合并
for metric in "${METRICS[@]}"; do
  echo ""
  echo "Processing metric: ${metric}"

  output_file="${METRICS_ROOT_DIR}/${metric}.csv"
  temp_file="${METRICS_ROOT_DIR}/.temp_${metric}.csv"

  methods_header=$(IFS=,; echo "${METHODS[*]}")
  echo "dataset,${methods_header}" > "${temp_file}"

  for dataset in "${DATASETS[@]}"; do
    row="${dataset}"

    for method in "${METHODS[@]}"; do
      values=()

      for seed_name in "${SEED_NAMES[@]}"; do
        csv_file="${METRICS_ROOT_DIR}/${seed_name}/${dataset}.${method}.${metric}.csv"
        if [[ -f "${csv_file}" ]]; then
          value=$(awk -F',' 'NR==2 {print $1}' "${csv_file}")
          if [[ -n "${value}" ]] && is_number "${value}"; then
            values+=("${value}")
          fi
        fi
      done

      if [[ ${#values[@]} -gt 0 ]]; then
        avg_value=$(printf "%s\n" "${values[@]}" | awk '{sum+=$1; cnt+=1} END {if (cnt>0) printf "%.10f", sum/cnt}')
        row="${row},${avg_value}"
      else
        row="${row},N/A"
      fi
    done

    echo "${row}" >> "${temp_file}"
  done

  mv "${temp_file}" "${output_file}"
  echo "✓ Saved: ${output_file}"
done

echo ""
echo "===== CSV merging completed ====="
