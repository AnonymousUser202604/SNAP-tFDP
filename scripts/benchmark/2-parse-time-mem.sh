#!/usr/bin/env bash
set -euo pipefail

# 所有方法列表
METHODS=(
  "pmds"
  "snap_tfdp"
  "snap_tfdp_par"
  "snap_tfdp_gpu"
  "drgraph"
  "sfdp"
  "fr"
  "fa2"
  "linlog"
  "tsnet"
  "tfdp_ibfft_cpu"
)

# 参数配置
RESULT_BASE_DIR="./results/benchmark"
STATISTICS_OUTPUT_DIR="./statistics/benchmark"

# 确保输出目录存在
mkdir -p "${STATISTICS_OUTPUT_DIR}"

# 解析时间的函数
parse_time() {
  local method="$1"
  local log_file="$2"
  local time_seconds=""

  case "${method}" in
    fr)
      # Elapsed (wall clock) time (h:mm:ss or m:ss): 1:57.32 or 2:25:31
      time_seconds=$(grep -oP "Elapsed \(wall clock\) time.*?:\s*\K[0-9:.]+" "${log_file}" 2>/dev/null || echo "")
      if [[ -n "${time_seconds}" ]]; then
        # 转换 h:mm:ss 格式为秒
        if [[ "${time_seconds}" =~ ^([0-9]+):([0-9]+):([0-9]+)$ ]]; then
          local hour="${BASH_REMATCH[1]}"
          local min="${BASH_REMATCH[2]}"
          local sec="${BASH_REMATCH[3]}"
          time_seconds=$(awk "BEGIN {printf \"%.6f\", ${hour} * 3600 + ${min} * 60 + ${sec}}")
        # 转换 m:ss.xx 格式为秒
        elif [[ "${time_seconds}" =~ ^([0-9]+):([0-9]+)\.([0-9]+)$ ]]; then
          local min="${BASH_REMATCH[1]}"
          local sec="${BASH_REMATCH[2]}"
          local frac="${BASH_REMATCH[3]}"
          time_seconds=$(awk "BEGIN {printf \"%.6f\", ${min} * 60 + ${sec} + 0.${frac}}")
        fi
      fi
      ;;
    pmds)
      # pivotMDS time: 17287 ms
      time_seconds=$(grep -oP "pivotMDS time:\s*\K[0-9.]+" "${log_file}" 2>/dev/null || echo "")
      if [[ -n "${time_seconds}" ]]; then
        # 转换毫秒为秒
        time_seconds=$(awk "BEGIN {printf \"%.6f\", ${time_seconds} / 1000}")
      fi
      ;;
    sfdp)
      # Elapsed (wall clock) time (h:mm:ss or m:ss): 0:05.93 or 2:17:09
      time_seconds=$(grep -oP "Elapsed \(wall clock\) time.*?:\s*\K[0-9:.]+" "${log_file}" 2>/dev/null || echo "")
      if [[ -n "${time_seconds}" ]]; then
        # 转换 h:mm:ss 格式为秒
        if [[ "${time_seconds}" =~ ^([0-9]+):([0-9]+):([0-9]+)$ ]]; then
          local hour="${BASH_REMATCH[1]}"
          local min="${BASH_REMATCH[2]}"
          local sec="${BASH_REMATCH[3]}"
          time_seconds=$(awk "BEGIN {printf \"%.6f\", ${hour} * 3600 + ${min} * 60 + ${sec}}")
        # 转换 m:ss.xx 格式为秒
        elif [[ "${time_seconds}" =~ ^([0-9]+):([0-9]+)\.([0-9]+)$ ]]; then
          local min="${BASH_REMATCH[1]}"
          local sec="${BASH_REMATCH[2]}"
          local frac="${BASH_REMATCH[3]}"
          time_seconds=$(awk "BEGIN {printf \"%.6f\", ${min} * 60 + ${sec} + 0.${frac}}")
        fi
      fi
      ;;
    linlog)
      # layout time: 48.227940 s
      time_seconds=$(grep -oP "layout time:\s*\K[0-9.]+" "${log_file}" 2>/dev/null || echo "")
      if [[ -n "${time_seconds}" ]]; then
        time_seconds=$(awk "BEGIN {printf \"%.6f\", ${time_seconds}}")
      fi
      ;;
    fa2)
      # layout time: 103.4 s
      time_seconds=$(grep -oP "layout time:\s*\K[0-9.]+" "${log_file}" 2>/dev/null || echo "")
      if [[ -n "${time_seconds}" ]]; then
        time_seconds=$(awk "BEGIN {printf \"%.6f\", ${time_seconds}}")
      fi
      ;;
    tsnet)
      # tsNET running time: 208.873983 s
      time_seconds=$(grep -oP "tsNET running time:\s*\K[0-9.]+" "${log_file}" 2>/dev/null || echo "")
      if [[ -n "${time_seconds}" ]]; then
        time_seconds=$(awk "BEGIN {printf \"%.6f\", ${time_seconds}}")
      fi
      ;;
    drgraph)
      # [ALL] CPU Time: 2433.604 s Real Time: 2434.841 s
      time_seconds=$(grep -oP "\[ALL\].*?Real Time:\s*\K[0-9.]+" "${log_file}" 2>/dev/null || echo "")
      if [[ -n "${time_seconds}" ]]; then
        time_seconds=$(awk "BEGIN {printf \"%.6f\", ${time_seconds}}")
      fi
      ;;
    drgraph_p16)
      # [ALL] CPU Time: 2433.604 s Real Time: 2434.841 s
      time_seconds=$(grep -oP "\[ALL\].*?Real Time:\s*\K[0-9.]+" "${log_file}" 2>/dev/null || echo "")
      if [[ -n "${time_seconds}" ]]; then
        time_seconds=$(awk "BEGIN {printf \"%.6f\", ${time_seconds}}")
      fi
      ;;
    tfdp)
      # Optimization time: 2.735 seconds
      time_seconds=$(grep -oP "Optimization time:\s*\K[0-9.]+" "${log_file}" 2>/dev/null || echo "")
      if [[ -n "${time_seconds}" ]]; then
        time_seconds=$(awk "BEGIN {printf \"%.6f\", ${time_seconds}}")
      fi
      ;;
    tfdp_ibfft_cpu)
      # Optimization time: 2.735 seconds
      time_seconds=$(grep -oP "Optimization time:\s*\K[0-9.]+" "${log_file}" 2>/dev/null || echo "")
      if [[ -n "${time_seconds}" ]]; then
        time_seconds=$(awk "BEGIN {printf \"%.6f\", ${time_seconds}}")
      fi
      ;;
    tfdp_ibfft_gpu)
      # Optimization time: 2.735 seconds
      time_seconds=$(grep -oP "Optimization time:\s*\K[0-9.]+" "${log_file}" 2>/dev/null || echo "")
      if [[ -n "${time_seconds}" ]]; then
        time_seconds=$(awk "BEGIN {printf \"%.6f\", ${time_seconds}}")
      fi
      ;;
    nsgl)
      # Runtime: 1.11799 s
      time_seconds=$(grep -oP "Runtime:\s*\K[0-9.]+" "${log_file}" 2>/dev/null || echo "")
      if [[ -n "${time_seconds}" ]]; then
        time_seconds=$(awk "BEGIN {printf \"%.6f\", ${time_seconds}}")
      fi
      ;;
    nsgl_p16)
      # Runtime: 1.11799 s
      time_seconds=$(grep -oP "Runtime:\s*\K[0-9.]+" "${log_file}" 2>/dev/null || echo "")
      if [[ -n "${time_seconds}" ]]; then
        time_seconds=$(awk "BEGIN {printf \"%.6f\", ${time_seconds}}")
      fi
      ;;
    nsgl_par)
      # Runtime: 1.11799 s
      time_seconds=$(grep -oP "Runtime:\s*\K[0-9.]+" "${log_file}" 2>/dev/null || echo "")
      if [[ -n "${time_seconds}" ]]; then
        time_seconds=$(awk "BEGIN {printf \"%.6f\", ${time_seconds}}")
      fi
      ;;
    nsgl_gpu)
      # Runtime: 1.11799 s
      time_seconds=$(grep -oP "Runtime:\s*\K[0-9.]+" "${log_file}" 2>/dev/null || echo "")
      if [[ -n "${time_seconds}" ]]; then
        time_seconds=$(awk "BEGIN {printf \"%.6f\", ${time_seconds}}")
      fi
      ;;
    ogdf_pmds)
      # PMDS time: 2.055 s
      time_seconds=$(grep -oP "PMDS time:\s*\K[0-9.]+" "${log_file}" 2>/dev/null || echo "")
      if [[ -n "${time_seconds}" ]]; then
        time_seconds=$(awk "BEGIN {printf \"%.6f\", ${time_seconds}}")
      fi
      ;;
  esac

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

echo "===== Parsing time and memory for all methods ====="
echo ""

# 临时文件存储所有数据
declare -A time_data
declare -A mem_data
declare -a datasets

# 遍历每个方法，收集数据
for method in "${METHODS[@]}"; do
  echo "===== Processing method=${method} ====="

  # 用于存储该方法的所有运行数据
  declare -A method_time_values
  declare -A method_mem_values

  # 扫描所有 seed 目录
  for seed_dir in "${RESULT_BASE_DIR}"/__seed*; do
    if [[ -d "${seed_dir}" ]]; then
      INPUT_DIR="${seed_dir}/${method}"

      if [[ -d "${INPUT_DIR}" ]]; then
        # 遍历该方法目录下的所有 .log 文件
        for log_file in "${INPUT_DIR}"/*.log; do
          if [[ -f "${log_file}" ]]; then
            dataset=$(basename "${log_file}" .log)
            time_sec=$(parse_time "${method}" "${log_file}")
            mem_mb=$(parse_memory "${log_file}")

            # 处理空值
            [[ -z "${time_sec}" ]] && time_sec="N/A"
            [[ -z "${mem_mb}" ]] && mem_mb="N/A"

            # 存储数据到临时数组
            if [[ -z "${method_time_values[${dataset}]:-}" ]]; then
              method_time_values["${dataset}"]="${time_sec}"
              method_mem_values["${dataset}"]="${mem_mb}"
            else
              method_time_values["${dataset}"]="${method_time_values[${dataset}]} ${time_sec}"
              method_mem_values["${dataset}"]="${method_mem_values[${dataset}]} ${mem_mb}"
            fi

            # 记录数据集名（去重）
            if [[ ! " ${datasets[@]} " =~ " ${dataset} " ]]; then
              datasets+=("${dataset}")
            fi
          fi
        done
      fi
    fi
  done

  # 也扫描不带 seed 的目录（兼容旧格式）
  INPUT_DIR="${RESULT_BASE_DIR}/${method}"
  if [[ -d "${INPUT_DIR}" ]]; then
    for log_file in "${INPUT_DIR}"/*.log; do
      if [[ -f "${log_file}" ]]; then
        dataset=$(basename "${log_file}" .log)
        time_sec=$(parse_time "${method}" "${log_file}")
        mem_mb=$(parse_memory "${log_file}")

        # 处理空值
        [[ -z "${time_sec}" ]] && time_sec="N/A"
        [[ -z "${mem_mb}" ]] && mem_mb="N/A"

        # 存储数据到临时数组
        if [[ -z "${method_time_values[${dataset}]:-}" ]]; then
          method_time_values["${dataset}"]="${time_sec}"
          method_mem_values["${dataset}"]="${mem_mb}"
        else
          method_time_values["${dataset}"]="${method_time_values[${dataset}]} ${time_sec}"
          method_mem_values["${dataset}"]="${method_mem_values[${dataset}]} ${mem_mb}"
        fi

        # 记录数据集名（去重）
        if [[ ! " ${datasets[@]} " =~ " ${dataset} " ]]; then
          datasets+=("${dataset}")
        fi
      fi
    done
  fi

  # 对每个数据集的多次运行求均值
  for dataset in "${!method_time_values[@]}"; do
    time_values="${method_time_values[${dataset}]}"
    mem_values="${method_mem_values[${dataset}]}"

    # 计算时间均值
    if [[ "${time_values}" != "N/A" ]]; then
      avg_time=$(echo "${time_values}" | awk '{
        sum=0; count=0; valid=0
        for(i=1; i<=NF; i++) {
          if($i != "N/A") {
            sum+=$i; valid++
          }
        }
        if(valid > 0) printf "%.6f", sum/valid
        else print "N/A"
      }')
      [[ -z "${avg_time}" ]] && avg_time="N/A"
    else
      avg_time="N/A"
    fi

    # 计算内存均值
    if [[ "${mem_values}" != "N/A" ]]; then
      avg_mem=$(echo "${mem_values}" | awk '{
        sum=0; count=0; valid=0
        for(i=1; i<=NF; i++) {
          if($i != "N/A") {
            sum+=$i; valid++
          }
        }
        if(valid > 0) printf "%.3f", sum/valid
        else print "N/A"
      }')
      [[ -z "${avg_mem}" ]] && avg_mem="N/A"
    else
      avg_mem="N/A"
    fi

    # 存储最终数据
    time_data["${dataset},${method}"]="${avg_time}"
    mem_data["${dataset},${method}"]="${avg_mem}"
  done

  unset method_time_values method_mem_values
  echo "✓ Processed method: ${method}"
  echo ""
done

# 排序数据集名
IFS=$'\n' sorted_datasets=($(sort <<<"${datasets[*]}"))
unset IFS

# 生成 time.csv
echo "Generating time.csv..."
TIME_FILE="${STATISTICS_OUTPUT_DIR}/time.csv"
{
  # 写入表头（方法名）
  echo -n "dataset"
  for method in "${METHODS[@]}"; do
    echo -n ",${method}"
  done
  echo ""

  # 写入数据行
  for dataset in "${sorted_datasets[@]}"; do
    echo -n "${dataset}"
    for method in "${METHODS[@]}"; do
      value="${time_data[${dataset},${method}]:-N/A}"
      echo -n ",${value}"
    done
    echo ""
  done
} > "${TIME_FILE}"
echo "✓ Saved: ${TIME_FILE}"

# 生成 mem.csv
echo "Generating mem.csv..."
MEM_FILE="${STATISTICS_OUTPUT_DIR}/mem.csv"
{
  # 写入表头（方法名）
  echo -n "dataset"
  for method in "${METHODS[@]}"; do
    echo -n ",${method}"
  done
  echo ""

  # 写入数据行
  for dataset in "${sorted_datasets[@]}"; do
    echo -n "${dataset}"
    for method in "${METHODS[@]}"; do
      value="${mem_data[${dataset},${method}]:-N/A}"
      echo -n ",${value}"
    done
    echo ""
  done
} > "${MEM_FILE}"
echo "✓ Saved: ${MEM_FILE}"

echo ""
echo "===== All done! ====="
