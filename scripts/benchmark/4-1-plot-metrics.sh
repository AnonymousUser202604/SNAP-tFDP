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
  "pmds"
  "fr"
  "sfdp"
  "linlog"
  "fa2"
  "tsnet"
  "drgraph"
  "tfdp"
  "nsgl"
)

# 指标列表
METRICS=(
  "np2"
  "si"
  "ari"
)

# 指标显示名称
METRIC_NAMES=(
  "NP"
  "SI"
  "CQ"
)

# 参数配置
METRICS_DIR="./statistics/benchmark"
OUTPUT="./figures/benchmark/heatmap_metrics.svg"
DPI=300

# 构建 JSON 格式的参数
DATASETS_JSON=$(printf '%s\n' "${DATASETS[@]}" | jq -R . | jq -s .)
METHODS_JSON=$(printf '%s\n' "${METHODS[@]}" | jq -R . | jq -s .)
METRICS_JSON=$(printf '%s.csv\n' "${METRICS[@]}" | jq -R . | jq -s .)
METRIC_NAMES_JSON=$(printf '%s\n' "${METRIC_NAMES[@]}" | jq -R . | jq -s .)

echo "绘制指标热力图..."
python scripts/benchmark/plot_metrics.py \
    --metrics-dir "$METRICS_DIR" \
    --metric-files "$METRICS_JSON" \
    --metric-names "$METRIC_NAMES_JSON" \
    --datasets "$DATASETS_JSON" \
    --methods "$METHODS_JSON" \
    --output "$OUTPUT" \
    --dpi "$DPI"

echo "完成！"
