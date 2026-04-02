"""
从多个指标文件绘制热力图（子图数量自动根据指标文件确定）。
数据格式：CSV文件，第一列为数据集名，其余列为方法名
"""
import argparse
import json
import math
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.gridspec import GridSpec
import matplotlib as mpl

plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.family'] = 'Arial'

# 默认参数
DEFAULT_METRICS_DIR = './metrics/'
DEFAULT_METRIC_FILES = [
    'np2.csv',
    'si.csv',
    'ari.csv',
]

DEFAULT_METRIC_NAMES = [
    "NP",
    'SI',
    'CQ',
]

DEFAULT_DATASET_FILTER = [
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

DEFAULT_METHOD_FILTER = [
    "pmds",
    "fr",
    "sfdp",
    "linlog",
    "fa2",
    "tsnet",
    "drgraph",
    "tfdp",
    "nsgl",
]

# 方法名称映射
DEFAULT_METHOD_NAME_MAP = {
    "fr": "FR",
    "pmds": "PMDS",
    "sfdp": "SFDP",
    "linlog": "LinLog",
    "fa2": "FA2",
    "tsnet": "tsNET",
    "drgraph": "DRGraph",
    "tfdp": "t-FDP",
    "nsgl": "SNAP-tFDP",
    "nsgl_par": "SNAP-tFDP-Par",
    "nsgl_gpu": "SNAP-tFDP-GPU"
}


def read_metric_file(filepath, dataset_filter=None, method_filter=None):
    """
    读取指标文件。
    参数:
    - filepath: 文件路径
    - dataset_filter: 数据集筛选列表，None 表示使用所有数据集
    - method_filter: 方法筛选列表，None 表示使用所有方法
    返回: (matrix, methods, datasets)
    - matrix: shape (n_datasets, n_methods)，数值矩阵
    - methods: 方法名列表，对应列
    - datasets: 数据集名列表，对应行
    """
    df = pd.read_csv(filepath, index_col=0)

    # 应用数据集筛选并按筛选列表顺序排列
    if dataset_filter is not None:
        df = df.loc[df.index.isin(dataset_filter)]
        # 按 dataset_filter 中的顺序重新排列
        df = df.reindex([d for d in dataset_filter if d in df.index])

    # 应用方法筛选并按筛选列表顺序排列
    if method_filter is not None:
        df = df[df.columns.intersection(method_filter)]
        # 按 method_filter 中的顺序重新排列
        df = df[[m for m in method_filter if m in df.columns]]

    datasets = df.index.tolist()
    methods = df.columns.tolist()
    matrix = df.values
    return matrix, methods, datasets


def normalize_by_row(matrix):
    """
    按行归一化：每行（每个数据集）独立归一化到 [0, 1]
    最小值->0（紫色），最大值->1（绿色）
    """
    matrix = np.array(matrix, dtype=float)
    n_rows, n_cols = matrix.shape
    normalized = np.zeros_like(matrix)

    for row_idx in range(n_rows):
        row = matrix[row_idx, :]
        valid = np.isfinite(row)
        if not np.any(valid):
            normalized[row_idx, :] = 0.5
            continue
        r_min, r_max = np.nanmin(row), np.nanmax(row)
        if r_max != r_min:
            normalized[row_idx, :] = np.where(valid, (row - r_min) / (r_max - r_min), 0.5)
        else:
            normalized[row_idx, :] = 0.5

    return normalized


def main():
    parser = argparse.ArgumentParser(description='绘制指标热力图')
    parser.add_argument('--metrics-dir', type=str, default=DEFAULT_METRICS_DIR,
                        help='指标文件目录')
    parser.add_argument('--metric-files', type=str, default=None,
                        help='指标文件名列表（JSON格式或逗号分隔）')
    parser.add_argument('--metric-names', type=str, default=None,
                        help='指标名称列表（JSON格式或逗号分隔）')
    parser.add_argument('--datasets', type=str, default=None,
                        help='数据集筛选列表（JSON格式或逗号分隔）')
    parser.add_argument('--methods', type=str, default=None,
                        help='方法筛选列表（JSON格式或逗号分隔）')
    parser.add_argument('--method-map', type=str, default=None,
                        help='方法名称映射（JSON格式）')
    parser.add_argument('--output', type=str, default='./figures/heatmap_metrics.svg',
                        help='输出文件路径')
    parser.add_argument('--dpi', type=int, default=300,
                        help='输出DPI')

    args = parser.parse_args()

    # 解析参数
    metrics_dir = args.metrics_dir

    # 解析列表参数
    def parse_list_arg(arg, default):
        if arg is None:
            return default
        try:
            return json.loads(arg)
        except json.JSONDecodeError:
            return [x.strip() for x in arg.split(',')]

    metric_files = parse_list_arg(args.metric_files, DEFAULT_METRIC_FILES)
    metric_names = parse_list_arg(args.metric_names, DEFAULT_METRIC_NAMES)
    dataset_filter = parse_list_arg(args.datasets, DEFAULT_DATASET_FILTER)
    method_filter = parse_list_arg(args.methods, DEFAULT_METHOD_FILTER)

    # 解析方法名称映射
    if args.method_map:
        try:
            method_name_map = json.loads(args.method_map)
        except json.JSONDecodeError:
            method_name_map = DEFAULT_METHOD_NAME_MAP
    else:
        method_name_map = DEFAULT_METHOD_NAME_MAP

    # 读取所有指标数据
    matrices = []
    metric_names_found = []
    methods = None
    datasets = None

    for fname, metric_name in zip(metric_files, metric_names):
        filepath = os.path.join(metrics_dir, fname)
        if not os.path.isfile(filepath):
            print(f"文件不存在，跳过: {filepath}")
            continue

        matrix, m, d = read_metric_file(filepath, dataset_filter=dataset_filter, method_filter=method_filter)
        matrices.append(matrix)
        metric_names_found.append(metric_name)

        if methods is None:
            methods = m
            datasets = d

    n = len(matrices)
    if n == 0:
        print("错误：没有找到任何指标文件")
        return

    # 根据数据集和方法数量自适应图表大小
    n_datasets = len(datasets)
    n_methods = len(methods)
    cell_width = 1.5  # 每个单元格的宽度（英寸）
    cell_height = 1.5 / 2.5  # 每个单元格的高度（英寸），增加高度以改善垂直居中效果

    # 计算热力图部分的高度（数据行 + 均值行）
    heatmap_height = (n_datasets + 1) * cell_height
    fig_width = n_methods * cell_width * n + 1.5
    fig_height = heatmap_height + 0.2  # 增加高度以容纳下方标题

    # 基于单元格尺寸自适应字体大小
    dpi = args.dpi
    cell_width_px = cell_width * dpi
    cell_height_px = cell_height * dpi

    # 单元格内注释字体大小（基于高度，留出边距）
    annot_fontsize = max(6, int(cell_height_px * 0.14))

    # 坐标轴标签字体大小（基于宽度）
    tick_fontsize = max(7, int(cell_height_px * 0.12))

    # 标题字体大小
    title_fontsize = int(tick_fontsize * 1.5)

    # 使用 GridSpec 创建布局，每个子图之间有间隔
    fig = plt.figure(figsize=(fig_width, fig_height))
    gs = GridSpec(1, n, figure=fig, wspace=0.03)
    axes = [fig.add_subplot(gs[0, i]) for i in range(n)]

    for i, (matrix, metric_name, ax) in enumerate(zip(matrices, metric_names_found, axes)):

        # 添加均值行：只有当该列全为正常值时才计算平均值
        mean_row = np.full(matrix.shape[1], np.nan)
        for col_idx in range(matrix.shape[1]):
            col = matrix[:, col_idx]
            if np.all(np.isfinite(col)):  # 该列全为正常值
                mean_row[col_idx] = np.mean(col)
            # 否则保持为 nan，显示为 N/A

        matrix_with_mean = np.vstack([matrix, mean_row])
        datasets_with_mean = datasets + ['Mean']

        normalized = normalize_by_row(matrix_with_mean)

        annot_matrix = np.where(
            np.isfinite(matrix_with_mean),
            np.vectorize(lambda x: f'{x:.3f}')(matrix_with_mean),
            '-'
        )

        cmap = plt.cm.PiYG.copy()
        cmap.set_bad('white')  # N/A 格子显示白色背景

        # 映射方法名称用于显示
        display_methods = [method_name_map.get(m, m) for m in methods]

        # 只有第一个子图显示 y 轴标签
        sns.heatmap(
            normalized,
            annot=annot_matrix,
            fmt='',
            cmap=cmap,
            vmin=0,
            vmax=1,
            cbar=False,
            xticklabels=display_methods,
            yticklabels=datasets_with_mean if i == 0 else [],
            linewidths=0.8,
            linecolor='white',
            annot_kws={
                'fontsize': annot_fontsize,
                'weight': 'normal',
                'fontname': 'Arial',
                'ha': 'center',
                'va': 'center'
            },
            ax=ax
        )

        # 调整所有注释文本的对齐方式
        for text in ax.texts:
            text.set_ha('center')
            text.set_va('center')

        # 列名放到上方
        ax.set_xticklabels(display_methods, rotation=45, ha='center', fontsize=tick_fontsize, family='Arial')
        ax.xaxis.tick_top()
        ax.xaxis.set_label_position('top')

        # 增加刻度线宽度
        ax.tick_params(axis='both', which='major', width=1.5, length=6)

        ax.set_xlabel('')
        ax.set_ylabel('')

        if i == 0:
            yticklabels = ax.get_yticklabels()
            # 让 Mean 标签加粗并增大
            for j, label in enumerate(yticklabels):
                if label.get_text() == 'Mean':
                    label.set_weight('bold')
                    label.set_fontsize(tick_fontsize * 1.5)
                else:
                    label.set_fontsize(tick_fontsize)
                label.set_family('Arial')
            ax.set_yticklabels(yticklabels, family='Arial')

        # 在均值行上方添加分隔线（白色加粗）
        ax.axhline(y=n_datasets, color='white', linewidth=8, linestyle='-')

        # 标题移到下方（使用 text 而不是 set_title）
        ax.text(0.5, -0.02, f'({chr(97 + i)}) {metric_name}',
                transform=ax.transAxes, ha='center', va='top',
                fontsize=title_fontsize, family='Arial', weight='normal')

    # 添加共享 colorbar
    cbar_ax = fig.add_axes([0.91, 0.15, 0.01, 0.7])
    sm = plt.cm.ScalarMappable(cmap='PiYG', norm=plt.Normalize(vmin=0, vmax=1))
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.ax.set_yticklabels([])  # 移除 y 轴刻度标签
    cbar.ax.yaxis.set_ticks([])  # 移除 y 轴刻度

    # 添加 Good 和 Bad 标签
    cbar_ax.text(0.5, 1.05, 'Good', ha='center', va='bottom', fontsize=title_fontsize, weight='regular',
                 transform=cbar_ax.transAxes, family='Arial')
    cbar_ax.text(0.5, -0.05, 'Bad', ha='center', va='top', fontsize=title_fontsize, weight='regular',
                 transform=cbar_ax.transAxes, family='Arial')

    # 保存图片
    output_path = args.output
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    print(f"已保存: {output_path}")

    plt.show()


if __name__ == '__main__':
    main()
