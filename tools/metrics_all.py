#!/usr/bin/env python3
"""
批量指标测试脚本
"""

import argparse
import re
import numpy as np
import pandas as pd
import subprocess
from pathlib import Path

# 默认配置
DEFAULT_DATA_DIR = Path("./data")
DEFAULT_RESULT_BASE_DIR = Path("./results")
DEFAULT_METRIC_OUTPUT_DIR = Path("./results/metrics")

DEFAULT_METHODS = [
    "fr",
    "sfdp",
    "linlog",
    "fa2",
    "tsnet",
    "drgraph",
    "tfdp",
    "nsgl_cpu_0_3",
    "nsgl_cpu_parallel_0_3",
    "nsgl_gpu_0_3",
]

DEFAULT_DATASETS = [
    "APH",
    "aircraft",
    "co_author_8391",
    "socfb-Yale4",
    "ACO",
    "socfb-UF21",
    "soc-Flickr-ASU",
    "com-dblp",
    "com-amazon",
    "com-youtube",
    "com-orkut",
    "com-lj",
]

DEFAULT_METRICS = [
    "cs",
    "icap",
    "qgg",
    "np",
    "ari",
    "si",
]


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="批量指标测试脚本")

    parser.add_argument(
        "--methods",
        type=str,
        default=None,
        help="方法列表，逗号分隔 (e.g., 'fr,fa2,linlog')"
    )
    parser.add_argument(
        "--datasets",
        type=str,
        default=None,
        help="数据集列表，逗号分隔 (e.g., 'APH,co_author_8391')"
    )
    parser.add_argument(
        "--metrics",
        type=str,
        default=None,
        help="指标列表，逗号分隔 (e.g., 'icap,ari')"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="数据目录路径"
    )
    parser.add_argument(
        "--result-base-dir",
        type=str,
        default=None,
        help="结果基础目录路径"
    )
    parser.add_argument(
        "--metrics-output-dir",
        type=str,
        default=None,
        help="指标输出目录路径"
    )

    return parser.parse_args()


def get_config():
    """获取配置，优先使用命令行参数，否则使用默认值"""
    args = parse_args()

    # 解析方法列表
    if args.methods:
        methods = [m.strip() for m in args.methods.split(",")]
    else:
        methods = DEFAULT_METHODS

    # 解析数据集列表
    if args.datasets:
        datasets = [d.strip() for d in args.datasets.split(",")]
    else:
        datasets = DEFAULT_DATASETS

    # 解析指标列表
    if args.metrics:
        metrics = [m.strip() for m in args.metrics.split(",")]
    else:
        metrics = DEFAULT_METRICS

    # 解析目录路径
    data_dir = Path(args.data_dir) if args.data_dir else DEFAULT_DATA_DIR
    result_base_dir = Path(args.result_base_dir) if args.result_base_dir else DEFAULT_RESULT_BASE_DIR
    metrics_output_dir = Path(args.metrics_output_dir) if args.metrics_output_dir else DEFAULT_METRIC_OUTPUT_DIR

    return {
        "methods": methods,
        "datasets": datasets,
        "metrics": metrics,
        "data_dir": data_dir,
        "result_base_dir": result_base_dir,
        "metrics_output_dir": metrics_output_dir,
    }


# 懒加载指标函数
def get_metric_func(metric_name):
    """懒加载指标计算函数"""
    if metric_name == 'ari':
        from metrics.ari import calculate_cq_ari
        return calculate_cq_ari
    elif metric_name == 'si':
        from metrics.si import calculate_silhouette
        return calculate_silhouette
    elif metric_name == 'cs':
        from metrics.cs import calculate_cluster_separation
        return calculate_cluster_separation
    elif metric_name == 'icap':
        from metrics.icap import calculate_icap
        return calculate_icap
    elif metric_name == 'cd':
        from metrics.cd import calculate_cluster_distance
        return calculate_cluster_distance
    elif metric_name == 'qgg':
        return None  # qgg 使用外部程序计算
    elif metric_name == 'np':
        return None  # np 使用外部程序计算
    else:
        raise ValueError(f"Unknown metric: {metric_name}")


def load_positions(filepath):
    """加载布局结果文件 (n行2列浮点数矩阵)"""
    pos = np.loadtxt(filepath, dtype=float)

    # 检查并处理 NaN
    nan_mask = np.isnan(pos)
    nan_count = np.sum(nan_mask)
    total_count = pos.size

    if nan_count > 0:
        nan_rate = (nan_count / total_count) * 100
        print(f"    [WARNING] Found {nan_count}/{total_count} NaN values ({nan_rate:.2f}%), replacing with 0.0")
        pos[nan_mask] = 0.0

    return pos


def load_labels(filepath):
    """加载标记文件 (n行1列整数)"""
    return np.loadtxt(filepath, dtype=int).flatten()


def load_edges(filepath):
    """
    加载图数据文件
    首行: n m (点数和边数)
    之后m行: u v w (起点、终点、边权，边权忽略)
    返回: edges ndarray (m, 2)
    """
    with open(filepath, 'r') as f:
        f.readline()  # 跳过首行 (n, m)
        edges = np.loadtxt(f, dtype=int, usecols=(0, 1))
    return edges


def calculate_qgg_external(edges_file, pos_file):
    """
    使用外部 C++ 程序计算 qgg 指标
    性能优化：直接调用编译的二进制程序，避免 Python 开销
    """
    try:
        # 查找 metrics_qgg 可执行文件
        script_dir = Path(__file__).parent
        qgg_bin = script_dir / "metrics" / "qgg"

        # 如果不存在，尝试 .exe 后缀（Windows）
        if not qgg_bin.exists():
            qgg_bin = script_dir / "metrics" / "qgg.exe"

        if not qgg_bin.exists():
            raise FileNotFoundError(f"qgg executable not found at {script_dir}/metrics/qgg")

        # 调用外部程序，只读取标准输出
        result = subprocess.run(
            [str(qgg_bin), str(edges_file), str(pos_file)],
            capture_output=True,
            text=True,
            timeout=7200
        )

        if result.returncode != 0:
            raise RuntimeError(f"qgg program failed: {result.stderr}")

        # 解析标准输出的第一行作为结果
        qgg_value = float(result.stdout.strip().split('\n')[0])
        return qgg_value

    except subprocess.TimeoutExpired:
        raise RuntimeError("qgg calculation timeout (>1 hour)")
    except Exception as e:
        raise RuntimeError(f"Failed to calculate qgg: {e}")


def calculate_np_external(edges_file, pos_file, ring_r=2):
    """
    使用外部 C++ 程序计算 np 指标
    性能优化：直接调用编译的二进制程序，避免 Python 开销
    """
    try:
        # 查找 metrics_np 可执行文件
        script_dir = Path(__file__).parent
        np_bin = script_dir / "metrics" / "np"

        # 如果不存在，尝试 .exe 后缀（Windows）
        if not np_bin.exists():
            np_bin = script_dir / "metrics" / "np.exe"

        if not np_bin.exists():
            raise FileNotFoundError(f"np executable not found at {script_dir}/metrics/np")

        # 调用外部程序，只读取标准输出
        result = subprocess.run(
            [str(np_bin), str(edges_file), str(pos_file), str(ring_r)],
            capture_output=True,
            text=True,
            timeout=3600  # 1小时超时
        )

        if result.returncode != 0:
            raise RuntimeError(f"np program failed: {result.stderr}")

        # 解析标准输出的第一行作为结果
        np_value = float(result.stdout.strip().split('\n')[0])
        return np_value

    except subprocess.TimeoutExpired:
        raise RuntimeError("np calculation timeout (>1 hour)")
    except Exception as e:
        raise RuntimeError(f"Failed to calculate np: {e}")


def calculate_metric(metric_name, pos, labels, edges, dataset, edges_file=None, pos_file=None):
    """计算指标"""
    default_label = -1

    if metric_name == 'ari':
        func = get_metric_func('ari')
        return func(
            node_positions=pos,
            ground_truth_labels=labels,
            n_clusters=None,
            default_label=default_label
        )
    elif metric_name == 'si':
        func = get_metric_func('si')
        return func(
            positions=pos,
            labels=labels,
            default_label=default_label
        )
    elif metric_name == 'cs':
        func = get_metric_func('cs')
        return func(
            positions=pos,
            labels=labels,
            default_label=default_label
        )
    elif metric_name == 'icap':
        func = get_metric_func('icap')
        return func(
            positions=pos,
            labels=labels,
            edges=edges,
            default_label=default_label
        )
    elif metric_name == 'cd':
        func = get_metric_func('cd')
        return func(
            positions=pos,
            labels=labels,
            mode='min',
            default_label=default_label
        )
    elif metric_name == 'qgg':
        # qgg 使用外部程序计算，需要文件路径
        if edges_file is None or pos_file is None:
            raise ValueError("qgg metric requires edges_file and pos_file parameters")
        return calculate_qgg_external(edges_file, pos_file)
    elif metric_name == 'np':
        # np 使用外部程序计算，需要文件路径，默认 ring_r=2
        if edges_file is None or pos_file is None:
            raise ValueError("np metric requires edges_file and pos_file parameters")
        return calculate_np_external(edges_file, pos_file, ring_r=2)
    elif metric_name.startswith('np') and len(metric_name) > 2 and metric_name[2:].isdigit():
        # np1, np2, np3, ... 格式
        ring_r = int(metric_name[2:])
        if edges_file is None or pos_file is None:
            raise ValueError(f"{metric_name} metric requires edges_file and pos_file parameters")
        return calculate_np_external(edges_file, pos_file, ring_r=ring_r)
    else:
        raise ValueError(f"Unknown metric: {metric_name}")


def main():
    # 获取配置
    config = get_config()

    data_dir = config["data_dir"]
    result_base_dir = config["result_base_dir"]
    metrics_output_dir = config["metrics_output_dir"]
    methods = config["methods"]
    datasets = config["datasets"]
    metrics = config["metrics"]

    # 创建输出目录
    metrics_output_dir.mkdir(parents=True, exist_ok=True)

    # 遍历所有指标
    for metric in metrics:
        print()
        print("=" * 50)
        print(f"Running metric: {metric}")
        print("=" * 50)

        for method in methods:
            print()
            print(f"--- Method: {method} ---")

            for dataset in datasets:
                # 收集该 dataset-method 组合的所有 iter-seed 结果
                iter_seed_results = []

                # 尝试从方法名中提取 iter 和 seed 信息
                match = re.search(r'iter_(\d+).*seed_(\d+)', method)
                if match:
                    iter_num = int(match.group(1))
                    seed_num = int(match.group(2))

                    # 构建文件路径
                    fp_pos = result_base_dir / f"{dataset}.iter_{iter_num}.seed_{seed_num}.layout.txt"
                    fp_labels = data_dir / f"{dataset}.attr"
                    fp_edges = data_dir / f"{dataset}.txt"

                    # 检查文件是否存在
                    if not fp_pos.exists():
                        print(f"  [SKIP] Position file not found: {fp_pos}")
                        continue

                    if not fp_labels.exists():
                        print(f"  [SKIP] Labels file not found: {fp_labels}")
                        continue

                    if not fp_edges.exists():
                        print(f"  [SKIP] Edges file not found: {fp_edges}")
                        continue

                    print(f"  [{dataset}] ", end="", flush=True)

                    try:
                        # 加载数据
                        pos = load_positions(fp_pos)
                        labels = load_labels(fp_labels)
                        edges = load_edges(fp_edges)

                        # 验证数据一致性
                        if len(pos) != len(labels):
                            print(f"ERROR: Position({len(pos)}) and labels({len(labels)}) mismatch")
                            continue

                        # 计算指标
                        if metric == 'qgg' or metric == 'np' or (metric.startswith('np') and len(metric) > 2 and metric[2:].isdigit()):
                            result = calculate_metric(metric, pos, labels, edges, dataset,
                                                      edges_file=fp_edges, pos_file=fp_pos)
                        else:
                            result = calculate_metric(metric, pos, labels, edges, dataset)

                        print(f"{metric}:{result}")
                        iter_seed_results.append((iter_num, seed_num, result))

                    except Exception as e:
                        print(f"ERROR: {e}")
                        continue
                else:
                    # 标准路径格式（不含 iter/seed）
                    fp_pos = result_base_dir / method / f"{dataset}.txt"
                    fp_labels = data_dir / f"{dataset}.attr"
                    fp_edges = data_dir / f"{dataset}.txt"

                    if not fp_pos.exists():
                        print(f"  [SKIP] Position file not found: {fp_pos}")
                        continue

                    if not fp_labels.exists():
                        print(f"  [SKIP] Labels file not found: {fp_labels}")
                        continue

                    if not fp_edges.exists():
                        print(f"  [SKIP] Edges file not found: {fp_edges}")
                        continue

                    print(f"  [{dataset}] ", end="", flush=True)

                    try:
                        pos = load_positions(fp_pos)
                        labels = load_labels(fp_labels)
                        edges = load_edges(fp_edges)

                        if len(pos) != len(labels):
                            print(f"ERROR: Position({len(pos)}) and labels({len(labels)}) mismatch")
                            continue

                        if metric == 'qgg' or metric == 'np' or (metric.startswith('np') and len(metric) > 2 and metric[2:].isdigit()):
                            result = calculate_metric(metric, pos, labels, edges, dataset,
                                                      edges_file=fp_edges, pos_file=fp_pos)
                        else:
                            result = calculate_metric(metric, pos, labels, edges, dataset)

                        print(f"{metric}:{result}")
                        iter_seed_results.append((None, None, result))

                    except Exception as e:
                        print(f"ERROR: {e}")
                        continue

                # 生成 CSV 文件：dataset.method.metric.csv
                if iter_seed_results:
                    output_file = metrics_output_dir / f"{dataset}.{method}.{metric}.csv"

                    # 构建 DataFrame
                    data = []
                    for iter_num, seed_num, value in iter_seed_results:
                        if iter_num is not None:
                            data.append({"iter": iter_num, metric: value})
                        else:
                            data.append({metric: value})

                    df = pd.DataFrame(data)

                    # 如果有 iter 列，按 iter 排序
                    if "iter" in df.columns:
                        df = df.sort_values("iter")
                        # 只保留 iter 和 metric 两列
                        df = df[["iter", metric]]
                    else:
                        df = df[[metric]]

                    df.to_csv(output_file, index=False)
                    print(f"  Saved: {output_file}")

    print()
    print("=" * 50)
    print("All metrics completed!")
    print(f"Results directory: {metrics_output_dir}")
    print("=" * 50)


if __name__ == '__main__':
    main()
