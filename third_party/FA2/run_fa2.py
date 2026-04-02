import networkx as nx
from fa2_modified import ForceAtlas2
import sys
import time

forceatlas2 = ForceAtlas2(
    outboundAttractionDistribution=True,
    linLogMode=False,
    adjustSizes=False,
    edgeWeightInfluence=1.0,
    jitterTolerance=1.0,
    barnesHutOptimize=True,
    barnesHutTheta=1.2,
    multiThreaded=False,
    scalingRatio=2.0,
    strongGravityMode=False,
    gravity=1.0,
    verbose=True,
)


def main():
    if len(sys.argv) != 4:
        print("Usage: python run_fa2.py <edges_file> <init_pos_file> <output_pos_file>")
        sys.exit(1)

    edges_file, init_pos_file, output_file = sys.argv[1], sys.argv[2], sys.argv[3]

    # 读取图数据（每行 u v，跳过注释行）
    G = nx.Graph()
    with open(edges_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('%') or line.startswith('#'):
                continue
            parts = line.split()
            if len(parts) >= 2:
                u, v = int(parts[0]), int(parts[1])
                G.add_edge(u, v)

    # 读取初始坐标（每行 x y，按行号作为节点 id）
    initial_pos = {}
    num_nodes = 0
    with open(init_pos_file, 'r') as f:
        for node_id, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) >= 2:
                initial_pos[node_id] = (float(parts[0]), float(parts[1]))
                num_nodes = node_id + 1

    # 运行 FA2 布局并计时
    t0 = time.time()
    positions = forceatlas2.forceatlas2_networkx_layout(G, pos=initial_pos, iterations=300)
    elapsed = time.time() - t0
    print(f"layout time: {elapsed:.1f} s")

    # 输出结果（首行时间，之后每行 x y）
    with open(output_file, 'w') as f:
        for node_id in range(num_nodes):
            x, y = positions[node_id]
            f.write(f"{x:.6f} {y:.6f}\n")


if __name__ == '__main__':
    main()
