import numpy as np
from scipy.spatial.distance import cdist, pdist
from itertools import combinations


def calculate_cluster_distance(positions, labels, mode='min', default_label=-1):
    """
    计算 Cluster Distance (CD)

    参数:
    positions: (N, 2) 节点坐标矩阵
    labels: (N,) 节点标签数组
    mode: 'min' (最小距离, 推荐) 或 'avg' (平均距离)
    default_label: 缺省社区标签, 不纳入计算。ACO/APH/co_author_8391 为 -1, 其余为 0

    返回:
    float: CD 指标值
    """
    # 跳过缺省标签, 只考虑非缺省社区
    print(f'[dl={default_label}]', end='')
    unique_classes = np.unique(labels)
    unique_classes = unique_classes[unique_classes != default_label]
    n_classes = len(unique_classes)

    if n_classes < 2:
        print("ERROR")
        return 0.0

    # 1. 计算全图归一化因子 (Max Graph Distance)
    # 论文要求：dividing by the maximum distance in the graph
    # 即使输入坐标是 0-1 归一化的, 最大距离可能是 sqrt(2), 所以这一步是必须的
    # all_dists = pdist(positions, metric='euclidean')
    # max_graph_dist = np.max(all_dists) if all_dists.size > 0 else 1.0

    max_graph_dist = 1.0

    if max_graph_dist == 0:
        return 0.0

    # 2. 按类别分组节点索引
    cluster_indices = {cls: np.where(labels == cls)[0] for cls in unique_classes}

    pairwise_distances = []

    # 3. 遍历所有类别对 (Combinations)
    # 例如 5 个类, 会有 C(5,2) = 10 对
    for c1, c2 in combinations(unique_classes, 2):
        # 获取两个类别的坐标点
        pos1 = positions[cluster_indices[c1]]
        pos2 = positions[cluster_indices[c2]]

        # 计算两个点集之间的距离矩阵 (M x N)
        dists_matrix = cdist(pos1, pos2, metric='euclidean')

        # 归一化
        dists_matrix_norm = dists_matrix / max_graph_dist

        dist_val = 0.0
        if mode == 'min':
            # Min Linkage: 两个聚类之间最近点的距离 (衡量分离度 separation)
            dist_val = np.min(dists_matrix_norm)
        elif mode == 'avg':
            # Average Linkage: 两个聚类之间所有连线的平均距离
            dist_val = np.mean(dists_matrix_norm)

        pairwise_distances.append(dist_val)

    # 4. 计算所有对的平均值
    final_cd = np.mean(pairwise_distances)

    return final_cd
