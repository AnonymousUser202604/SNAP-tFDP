#!/usr/bin/env bash
set -euo pipefail

# 参数配置
RESULT_BASE_DIR="./results/param-k"
STATISTICS_OUTPUT_DIR="./statistics/param-k"

# 确保输出目录存在
mkdir -p "${STATISTICS_OUTPUT_DIR}"

# 解析时间的函数（nsgl 方法）
parse_time() {
  local log_file="$1"
  local time_seconds=""

  # Runtime: 1.11799 s
  time_seconds=$(grep -oP "Runtime:\s*\K[0-9.]+" "${log_file}" 2>/dev/null || echo "")
  if [[ -n "${time_seconds}" ]]; then
    time_seconds=$(awk "BEGIN {printf \"%.6f\", ${time_seconds}}")
  fi

  echo "${time_seconds}"
}

# 解析内存的函数
parse_memory() {
  local log_file="$1"
  local mem_kb=""
  local mem_mb=""

  # Maximum resident set size (kbytes): 227948
  mem_kb=$(grep -oP "Maximum resident set size \(kbytes\):\s*\K[0-9]+" "${log_file}" 2>/dev/null || echo "")
  if [[ -n "${mem_kb}" ]]; then
    mem_mb=$(awk "BEGIN {printf \"%.3f\", ${mem_kb} / 1024}")
  fi

  echo "${mem_mb}"
}

# 全局存储：dataset × k
declare -A time_table
declare -A mem_table
declare -a datasets
declare -a k_values

# 收集所有 k 值（目录名 k_<num>）
for k_dir in "${RESULT_BASE_DIR}"/k_*; do
  if [[ -d "${k_dir}" ]]; then
    k_name=$(basename "${k_dir}")
    k_num="${k_name#k_}"
    if [[ "${k_num}" =~ ^[0-9]+$ ]]; then
      k_values+=("${k_num}")
    fi
  fi
done

# 没有任何 k 目录则直接退出
if [[ ${#k_values[@]} -eq 0 ]]; then
  echo "⚠ No k_* directories found under ${RESULT_BASE_DIR}"
  exit 0
fi

# k 按数值排序
IFS=$'\n' k_values=($(printf "%s\n" "${k_values[@]}" | sort -n))
unset IFS

# 遍历每个 k，解析日志
for k in "${k_values[@]}"; do
  echo "===== Processing k=${k} ====="

  INPUT_DIR="${RESULT_BASE_DIR}/k_${k}"
  if [[ ! -d "${INPUT_DIR}" ]]; then
    echo "⚠ Directory not found: ${INPUT_DIR}"
    continue
  fi

  for log_file in "${INPUT_DIR}"/*.log; do
    if [[ -f "${log_file}" ]]; then
      dataset=$(basename "${log_file}" .log)
      time_sec=$(parse_time "${log_file}")
      mem_mb=$(parse_memory "${log_file}")

      [[ -z "${time_sec}" ]] && time_sec="N/A"
      [[ -z "${mem_mb}" ]] && mem_mb="N/A"

      time_table["${dataset},k_${k}"]="${time_sec}"
      mem_table["${dataset},k_${k}"]="${mem_mb}"

      if [[ ! " ${datasets[*]} " =~ " ${dataset} " ]]; then
        datasets+=("${dataset}")
      fi
    fi
  done

done

# dataset 排序
IFS=$'\n' sorted_datasets=($(printf "%s\n" "${datasets[@]}" | sort))
unset IFS

# 生成合并后的 time.csv
TIME_FILE="${STATISTICS_OUTPUT_DIR}/time.csv"
{
  echo -n "dataset"
  for k in "${k_values[@]}"; do
    echo -n ",k_${k}"
  done
  echo ""

  for dataset in "${sorted_datasets[@]}"; do
    echo -n "${dataset}"
    for k in "${k_values[@]}"; do
      value="${time_table[${dataset},k_${k}]:-N/A}"
      echo -n ",${value}"
    done
    echo ""
  done
} > "${TIME_FILE}"
echo "✓ Saved: ${TIME_FILE}"

# 生成合并后的 mem.csv
MEM_FILE="${STATISTICS_OUTPUT_DIR}/mem.csv"
{
  echo -n "dataset"
  for k in "${k_values[@]}"; do
    echo -n ",k_${k}"
  done
  echo ""

  for dataset in "${sorted_datasets[@]}"; do
    echo -n "${dataset}"
    for k in "${k_values[@]}"; do
      value="${mem_table[${dataset},k_${k}]:-N/A}"
      echo -n ",${value}"
    done
    echo ""
  done
} > "${MEM_FILE}"
echo "✓ Saved: ${MEM_FILE}"

echo "===== All done! ====="