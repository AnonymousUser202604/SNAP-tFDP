#!/usr/bin/env bash
set -euo pipefail

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

METHODS=(
  "k_1"
  "k_2"
  "k_3"
  "k_4"
  "k_5"
  "k_6"
  "k_7"
  "k_8"
  "k_9"
  "k_10"
)

METRICS=(
  "ari"
  "si"
  "np2"
)

DATA_DIR="./data"
RESULT_BASE_DIR="./results/param-k"
METRICS_OUTPUT_DIR="./statistics/param-k"

DATASETS_STR=$(IFS=,; echo "${DATASETS[*]}")
METHODS_STR=$(IFS=,; echo "${METHODS[*]}")
METRICS_STR=$(IFS=,; echo "${METRICS[*]}")

echo "===== Running metrics for param_k ====="
echo "Datasets: ${DATASETS_STR}"
echo "Methods: ${METHODS_STR}"
echo "Metrics: ${METRICS_STR}"
echo ""

python tools/metrics_all.py \
  --datasets "${DATASETS_STR}" \
  --methods "${METHODS_STR}" \
  --metrics "${METRICS_STR}" \
  --data-dir "${DATA_DIR}" \
  --result-base-dir "${RESULT_BASE_DIR}" \
  --metrics-output-dir "${METRICS_OUTPUT_DIR}"

DATASETS_STR="${DATASETS_STR}" \
METHODS_STR="${METHODS_STR}" \
METRICS_STR="${METRICS_STR}" \
STATS_DIR="${METRICS_OUTPUT_DIR}" \
python - <<'PY'
import os
from pathlib import Path
import pandas as pd

datasets = [d.strip() for d in os.environ["DATASETS_STR"].split(",") if d.strip()]
methods = [m.strip() for m in os.environ["METHODS_STR"].split(",") if m.strip()]
metrics = [m.strip() for m in os.environ["METRICS_STR"].split(",") if m.strip()]
stats_dir = Path(os.environ["STATS_DIR"])

for metric in metrics:
    rows = []

    for dataset in datasets:
        row = {"dataset": dataset}

        for method in methods:
            split_file = stats_dir / f"{dataset}.{method}.{metric}.csv"
            value = ""

            if split_file.exists():
                df = pd.read_csv(split_file)
                if len(df) > 0:
                    if metric in df.columns:
                        value = df.loc[df.index[0], metric]
                    elif df.shape[1] > 0:
                        value = df.iloc[0, -1]

            row[method] = value
        rows.append(row)

    result_df = pd.DataFrame(rows, columns=["dataset", *methods])
    result_file = stats_dir / f"{metric}.result.csv"
    result_df.to_csv(result_file, index=False)
    print(f"Saved merged CSV: {result_file}")

    for dataset in datasets:
        for method in methods:
            split_file = stats_dir / f"{dataset}.{method}.{metric}.csv"
            if split_file.exists():
                split_file.unlink()
                print(f"Deleted split CSV: {split_file}")
PY

echo ""
echo "===== Metrics calculation completed ====="
echo "Merged files: ${METRICS_OUTPUT_DIR}/*.result.csv"
