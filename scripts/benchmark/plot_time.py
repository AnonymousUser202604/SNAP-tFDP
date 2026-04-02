"""
绘制数据集点数与方法运行时间的散点图。
横轴：边数 / 点数（M / N）或 边数 * 点数（M * N）
纵轴：运行时间 / 点数（Time / N）或 运行时间 * 点数（Time * N）
marker：不同方法用不同 marker 表示
"""
import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.family'] = 'Arial'

# 默认参数
DEFAULT_DATASET_INFO_PATH = './statistics/dataset_info.csv'
DEFAULT_TIME_PATH = './statistics/benchmark/time.csv'
DEFAULT_OUTPUT = './figures/benchmark/time_scatter.svg'
DEFAULT_DPI = 300

# 方法名称映射
DEFAULT_METHOD_DISPLAY_NAMES = {
    "ogdf_pmds": "PMDS",
    "sfdp": "SFDP",
    "drgraph": "DRGraph",
    "drgraph_p16": "DRGraph-P16",
    "tfdp_ibfft_cpu": "t-FDP",
    "tfdp_ibfft_gpu": "t-FDP-GPU",
    "nsgl": "SNAP-tFDP ",
    "nsgl_p16": "SNAP-tFDP-P16",
    "nsgl_gpu": "SNAP-tFDP-GPU",
}

# marker 和颜色映射
MARKERS = {
    "ogdf_pmds": "d",
    "sfdp": "*",
    "drgraph": ">",
    "drgraph_p16": "D",
    "tfdp_ibfft_cpu": "P",
    "tfdp_ibfft_gpu": "H",
    "nsgl": "s",
    "nsgl_p16": "o",
    "nsgl_gpu": "^",
}

COLORS = {
    "ogdf_pmds": "#af8981",
    "sfdp": "#e377c2",
    "drgraph": "#ff4500",
    "drgraph_p16": "#d62728",
    "tfdp_ibfft_cpu": "#4b0082",
    "tfdp_ibfft_gpu": "#bcbd22",
    "nsgl": "#3b3bed",
    "nsgl_p16": "#ffa90a",
    "nsgl_gpu": "#0a850a",
}


def plot_scatter(dataset_info, time_data, methods, method_display_names, markers, colors,
                 normalize=True, output_path=None, dpi=300):
    """
    绘制散点图
    normalize=True: 横纵轴都除以N（归一化）
    normalize=False: 横纵轴都乘以N（原始值）
    """
    # 创建图表
    fig, ax = plt.subplots(figsize=(8, 6))

    # 遍历每个方法
    for method in methods:
        if method not in time_data.columns:
            continue

        x_values = []  # 边数 / 点数 或 边数 * 点数
        y_values = []  # 时间 / 点数 或 时间 * 点数

        for dataset in time_data.index:
            if dataset not in dataset_info.index:
                continue

            n_nodes = dataset_info.loc[dataset, 'N']
            m_edges = dataset_info.loc[dataset, 'M']
            time_val = time_data.loc[dataset, method]

            # 跳过 N/A 值
            if pd.isna(time_val) or time_val == 'N/A':
                continue

            try:
                time_val = float(time_val)
            except (ValueError, TypeError):
                continue

            if time_val > 0 and n_nodes > 0:
                if normalize:
                    # 归一化：除以N
                    x_normalized = m_edges / n_nodes
                    y_normalized = time_val / n_nodes
                else:
                    # 原始值：直接使用 M 和 Time
                    x_normalized = m_edges
                    y_normalized = time_val

                x_values.append(x_normalized)
                y_values.append(y_normalized)

        if x_values:
            marker = markers.get(method, 'o')
            color = colors.get(method, 'black')
            display_name = method_display_names.get(method, method)

            ax.scatter(x_values, y_values, marker=marker, s=100, alpha=0.7,
                       label=display_name, color=color, edgecolors='black', linewidth=0.5)

            # 最小二乘法拟合趋势线
            if len(x_values) > 1:
                # 在对数空间中进行拟合
                log_x = np.log10(x_values)
                log_y = np.log10(y_values)

                # 一次多项式拟合（直线）
                coeffs = np.polyfit(log_x, log_y, 1)
                poly = np.poly1d(coeffs)

                # 生成拟合线的点
                x_fit = np.logspace(np.log10(min(x_values)), np.log10(max(x_values)), 100)
                log_x_fit = np.log10(x_fit)
                log_y_fit = poly(log_x_fit)
                y_fit = 10 ** log_y_fit

                # 绘制趋势线
                ax.plot(x_fit, y_fit, color=color, linestyle='-', linewidth=2, alpha=0.8)

            print(f"  {method}: {len(x_values)} 个数据点")

    # 设置对数尺度
    ax.set_xscale('log')
    ax.set_yscale('log')

    # 设置标签和标题
    if normalize:
        ax.set_xlabel('Edges per node)', fontsize=16, family='Arial')
        ax.set_ylabel('Time / N (seconds per node)', fontsize=16, family='Arial')
    else:
        ax.set_xlabel('Number of edges', fontsize=16, family='Arial')
        ax.set_ylabel('Time (seconds)', fontsize=16, family='Arial')

    # 添加网格
    ax.grid(True, alpha=0.3, linestyle='--', which='both')

    # 添加图例
    ax.legend(loc='upper left', fontsize=14, framealpha=0.9)

    # 调整布局
    plt.tight_layout()

    # 保存图片
    if output_path:
        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
        plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
        print(f"已保存: {output_path}")

    plt.close()


def main():
    parser = argparse.ArgumentParser(description='绘制时间散点图')
    parser.add_argument('--dataset-info', type=str, default=DEFAULT_DATASET_INFO_PATH,
                        help='数据集信息文件路径')
    parser.add_argument('--time-data', type=str, default=DEFAULT_TIME_PATH,
                        help='时间数据文件路径')
    parser.add_argument('--datasets', type=str, default=None,
                        help='数据集筛选列表（JSON格式或逗号分隔）')
    parser.add_argument('--methods', type=str, default=None,
                        help='方法筛选列表（JSON格式或逗号分隔）')
    parser.add_argument('--output', type=str, default=DEFAULT_OUTPUT,
                        help='输出文件路径（不含扩展名）')
    parser.add_argument('--dpi', type=int, default=DEFAULT_DPI,
                        help='输出DPI')

    args = parser.parse_args()

    # 解析参数
    def parse_list_arg(arg, default):
        if arg is None:
            return default
        try:
            import json
            return json.loads(arg)
        except (json.JSONDecodeError, ValueError):
            return [x.strip() for x in arg.split(',')]

    dataset_filter = parse_list_arg(args.datasets, None)
    method_filter = parse_list_arg(args.methods, None)

    # 读取数据
    print("读取数据集信息...")
    dataset_info = pd.read_csv(args.dataset_info, index_col=0)
    print(f"  数据集数: {len(dataset_info)}")

    print("读取时间数据...")
    time_data = pd.read_csv(args.time_data, index_col=0)
    print(f"  数据集数: {len(time_data)}, 方法数: {len(time_data.columns)}")

    # 应用筛选
    if dataset_filter is not None:
        time_data = time_data.loc[time_data.index.isin(dataset_filter)]
        dataset_info = dataset_info.loc[dataset_info.index.isin(dataset_filter)]
        print(f"  筛选后数据集数: {len(time_data)}")

    if method_filter is not None:
        time_data = time_data[[m for m in method_filter if m in time_data.columns]]
        print(f"  筛选后方法数: {len(time_data.columns)}")
        methods = method_filter
    else:
        methods = list(time_data.columns)

    # 打印各方法平均时间
    print("\n===== 各方法平均时间 =====")
    for method in methods:
        if method not in time_data.columns:
            continue
        col = pd.to_numeric(time_data[method], errors='coerce')
        valid = col.dropna()
        if len(valid) > 0:
            print(f"  {method}: {valid.mean():.6f} s  (共 {len(valid)} 个数据集)")
        else:
            print(f"  {method}: N/A")

    # 生成两张图
    # 图1：归一化（除以N）
    # print("\n绘制归一化图表...")
    # output_normalized = args.output.replace('.svg', '_normalized.svg')
    # plot_scatter(dataset_info, time_data, methods, DEFAULT_METHOD_DISPLAY_NAMES,
    #              MARKERS, COLORS, normalize=True, output_path=output_normalized, dpi=args.dpi)

    # 图2：原始值（乘以N）
    print("\n绘制原始值图表...")
    output_denormalized = args.output.replace('.svg', '_denormalized.svg')
    plot_scatter(dataset_info, time_data, methods, DEFAULT_METHOD_DISPLAY_NAMES,
                 MARKERS, COLORS, normalize=False, output_path=output_denormalized, dpi=args.dpi)

    print("\n完成！")


if __name__ == '__main__':
    main()
