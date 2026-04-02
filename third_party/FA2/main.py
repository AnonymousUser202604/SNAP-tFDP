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
    # 解析命令行参数
    # parser = argparse.ArgumentParser(description='Process a dataset with FA2 layout algorithm')
    # parser.add_argument('filename', type=str, 
    #                    help='Dataset filename')
    # args = parser.parse_args()
    
    # filename = args.filename

    filenames = ["com_orkut_sub","ogbn_arxiv"]

    # "ACO","APH","block_2000","co_author_8391","American75","Amherst41","Bowdoin47","Brandeis99","Brown11","Bucknell39","Carnegie49",
    #     "Colgate88","Dartmouth6","Duke14","Emory27","Georgetown15","Hamilton46","Haverford76","Howard90","Johns Hopkins55",
    #     "Lehigh96","MIT8","Maine59","Mich67","Middlebury45","Oberlin44","Pepperdine86","Princeton12","Reed98","Rochester38",
    #     "Santa74","Simmons81","Swarthmore42","Trinity100","Tufts18","Tulane29","UC64","UChicago30","USFCA72",
    #     "Vanderbilt48","Vassar85","Vermont70","Villanova62","Wake73","WashU32","Wellesley22","Wesleyan43","William77","Williams40","Yale4"

    # "lfr_n10000_mu0.42_s42","lfr_n20000_mu0.11_s42","lfr_n30000_mu0.24_s42","lfr_n40000_mu0.21_s42",
    # "lfr_n50000_mu0.47_s42","lfr_n60000_mu0.44_s42","lfr_n70000_mu0.55_s42","lfr_n80000_mu0.14_s42","lfr_n90000_mu0.31_s42","lfr_n100000_mu0.11_s42"
    # "BU10","Michigan23","Penn94"

    # "Auburn71","Cornell5","UCLA26","Maryland58","FSU53","Indiana69","UIllinois20","Texas80","MSU24","UF21","Texas84"

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
        # 记录开始时间
        time_start = time.time()

        positions = forceatlas2.forceatlas2_networkx_layout(G, pos=initial_pos, iterations=2000)
        
        time_total = time.time() - time_start
        
        mem_after_fa2 = get_memory_usage()
        peak_after_fa2 = get_peak_memory_usage()
        total_mem_increase = mem_after_fa2 - mem_before

        
        # 总结
        # print(f"\nMemory Summary for {filename}:")
        # print(f"  Initial: {mem_before/1024.0:.3f} MB")
        # print(f"  Total increase: {total_mem_increase/1024.0:.3f} MB")
        # print(f"  Peak memory: {peak_after_fa2/1024.0:.3f} MB")
        
        print(f"\nTotal time: {time_total:.3f} seconds")
        
        # 输出结果到result文件夹
        output_file = f'/home/sdu/wyf/graduate2026/result/2026_2_9_tmp/FA2/{filename}.txt'
        with open(output_file, 'w') as f:
            # 按节点id顺序输出(0, 1, 2, ...), 使用初始坐标文件的行数作为节点数
            f.write(f"{time_total:.6f}\n")
            for node_id in range(num_nodes_from_pos):
                x, y = positions[node_id]
                f.write(f"{x:.6f} {y:.6f}\n")

if __name__ == '__main__':
    main()