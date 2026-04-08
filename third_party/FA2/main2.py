import networkx as nx
from fa2_modified import ForceAtlas2
import os
import argparse
import sys
import time


def get_memory_usage():
    """获取当前内存使用量(VmRSS, 单位: KB)"""
    try:
        with open('/proc/self/status', 'r') as status_file:
            for line in status_file:
                if line.startswith('VmRSS:'):
                    parts = line.split()
                    if len(parts) >= 2:
                        return int(parts[1])  # VmRSS 单位是 KB
    except (IOError, ValueError) as e:
        print(f"Warning: Cannot read memory usage: {e}")
        return 0
    return 0


def get_peak_memory_usage():
    """获取峰值内存使用量(VmHWM, 单位: KB)"""
    try:
        with open('/proc/self/status', 'r') as status_file:
            for line in status_file:
                if line.startswith('VmHWM:'):
                    parts = line.split()
                    if len(parts) >= 2:
                        return int(parts[1])  # VmHWM 单位是 KB
    except (IOError, ValueError) as e:
        print(f"Warning: Cannot read peak memory usage: {e}")
        return 0
    return 0


# 初始化FA2布局算法
forceatlas2 = ForceAtlas2(
    # Behavior alternatives
    outboundAttractionDistribution=True,  # Dissuade hubs
    linLogMode=False,  # NOT IMPLEMENTED
    adjustSizes=False,  # Prevent overlap (NOT IMPLEMENTED)
    edgeWeightInfluence=1.0,

    # Performance
    jitterTolerance=1.0,  # Tolerance
    barnesHutOptimize=True,
    barnesHutTheta=1.2,
    multiThreaded=False,  # NOT IMPLEMENTED

    # Tuning
    scalingRatio=2.0,
    strongGravityMode=False,
    gravity=1.0,

    # Log
    verbose=True)


def main():
    filenames = ["Haverford76", "Simmons81", "Amherst41", "Pepperdine86", "Wesleyan43", "Bucknell39",
                 "Brandeis99", "Rochester38", "Johns Hopkins55", "APH", "WashU32", "Tulane29",
                 "co_author_8391", "Yale4", "ACO", "Cornell5", "Indiana69", "UF21", "ogbn_arxiv", "com_orkut_sub"]
    peak_after_fa2_list = []  # 保存每个数据的峰值内存 (数据名, 峰值KB)

    for filename in filenames:

        print(f"Processing {filename}...")
        # 初始内存状态
        mem_before = get_memory_usage()

        # 读取图数据
        # mtx_file = f'data/{filename}.mtx'
        # mtx_file = f'/home/sdu/wyf/graduate2026/data/{filename}.mtx'
        mtx_file = f'/home/sdu/wyf/graduate2026/data/facebook100_wyf/{filename}.mtx'
        if not os.path.exists(mtx_file):
            print(f"Error: File not found: {mtx_file}")
            sys.exit(1)

        G = nx.Graph()

        with open(mtx_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line:  # 跳过空行
                    parts = line.split()
                    if len(parts) >= 2:
                        u, v = int(parts[0]), int(parts[1])
                        G.add_edge(u, v)

        # 读取初始坐标
        # pos_file = f'/home/sdu/wyf/graduate2026/PMDS_init/pos_scale/{filename}.txt'
        pos_file = f'/home/sdu/wyf/graduate2026/PMDS_init/cpp_pos_scale/{filename}.txt'
        if not os.path.exists(pos_file):
            print(f"Error: File not found: {pos_file}")
            sys.exit(1)

        initial_pos = {}
        num_nodes_from_pos = 0

        with open(pos_file, 'r') as f:
            for node_id, line in enumerate(f):
                line = line.strip()
                if line:
                    parts = line.split()
                    if len(parts) >= 2:
                        x, y = float(parts[0]), float(parts[1])
                        initial_pos[node_id] = (x, y)
                        num_nodes_from_pos = node_id + 1

        # 运行FA2布局
        print("Running FA2 layout...")

        positions = forceatlas2.forceatlas2_networkx_layout(G, pos=initial_pos, iterations=2000)

        peak_after_fa2 = get_peak_memory_usage()
        peak_after_fa2_list.append((filename, peak_after_fa2))

        print(f"  Peak memory: {peak_after_fa2 / 1024.0:.3f} MB")

    # 将所有峰值写入 memory.txt：每行 数据名 峰值内存(MB)
    with open('memory.txt', 'w') as f:
        for name, peak_kb in peak_after_fa2_list:
            f.write(f"{name} {peak_kb / 1024.0:.3f}\n")
    print(f"Peak memory results written to memory.txt")


if __name__ == '__main__':
    main()
