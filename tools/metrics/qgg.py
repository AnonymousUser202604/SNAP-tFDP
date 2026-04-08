#!/usr/bin/env python3
import sys
import numpy as np
from gabrielgraph import build_gabriel_graph


# =========================
# 读取布局
# =========================
def load_layout(path):
    Y = np.loadtxt(path)
    if Y.ndim != 2 or Y.shape[1] != 2:
        raise ValueError("layout 必须是 n 行 2 列")
    return Y


# =========================
# 读取原图
# =========================
def load_graph(path):
    with open(path) as f:
        first = f.readline().split()
        N = int(first[0])
        M = int(first[1])

        adj = [set() for _ in range(N)]

        for line in f:
            if not line.strip():
                continue
            u, v, *_ = line.split()
            u = int(u)
            v = int(v)

            adj[u].add(v)
            adj[v].add(u)

    return adj


# =========================
# 从 GabrielGraph sparse 转邻接
# =========================
def gg_to_adj(gg_sparse):
    coo = gg_sparse.tocoo()
    N = gg_sparse.shape[0]

    adj = [set() for _ in range(N)]

    for u, v in zip(coo.row, coo.col):
        if u == v:
            continue
        adj[u].add(v)
        adj[v].add(u)

    return adj


# =========================
# JSS 计算
# =========================
def compute_jss(adj1, adj2):
    N = len(adj1)
    s = 0.0

    for u in range(N):

        A = adj1[u]
        B = adj2[u]

        if not A and not B:
            continue

        inter = len(A & B)
        union = len(A | B)

        s += inter / union

    return s


# =========================
# 主流程
# =========================
def main():
    if len(sys.argv) != 3:
        print("Usage:")
        print("python compute_jss_gg.py layout.txt graph.txt")
        return

    layout_path = sys.argv[1]
    graph_path = sys.argv[2]

    Y = load_layout(layout_path)
    N = Y.shape[0]

    # 原图邻接
    adj_gt = load_graph(graph_path)

    # 构建 GG
    ids = np.arange(N)
    gg_sparse = build_gabriel_graph(ids, Y, "adj-mat")

    adj_gg = gg_to_adj(gg_sparse)

    jss = compute_jss(adj_gt, adj_gg) / N

    print("JSS =", jss)


if __name__ == "__main__":
    main()
