#!/bin/bash

# 数据集列表
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

# 方法列表
METHODS=(
  "ogdf_pmds"
  "sfdp"
  "drgraph"
  "tfdp_ibfft_cpu"
  "nsgl"
  "tfdp_ibfft_gpu"
  "nsgl_gpu"
)

# 参数配置
DATASET_INFO="./statistics/dataset_info.csv"
TIME_DATA="./statistics/benchmark/time.csv"
OUTPUT="./figures/benchmark/time_scatter.svg"
DPI=300

# 构建 JSON 格式的参数
DATASETS_JSON=$(printf '%s\n' "${DATASETS[@]}" | jq -R . | jq -s .)
METHODS_JSON=$(printf '%s\n' "${METHODS[@]}" | jq -R . | jq -s .)

echo "绘制时间散点图..."
python scripts/benchmark/plot_time.py \
    --dataset-info "$DATASET_INFO" \
    --time-data "$TIME_DATA" \
    --datasets "$DATASETS_JSON" \
    --methods "$METHODS_JSON" \
    --output "$OUTPUT" \
    --dpi "$DPI"

echo "完成！"
