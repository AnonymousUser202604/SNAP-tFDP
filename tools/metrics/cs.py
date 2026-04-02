import numpy as np
from scipy.spatial.distance import cdist, pdist


def calculate_cluster_separation(positions, labels, default_label=-1):
    """
    计算 Cluster Separation (CS)
    公式: CS = d_intra / (d_intra + d_inter)
    缺省社区标签由 default_label 指定, 不参与计算。ACO/APH/co_author_8391 为 -1, 其余为 0。

    其中:
    - d_intra: 所有簇的平均半径 (Average Intra-cluster Radius)
    - d_inter: 所有簇质心之间的平均距离 (Average Inter-cluster Distance)
    """
    positions = np.asarray(positions)
    labels = np.asarray(labels)
    # 跳过缺省标签的节点（缺省社区不纳入指标计算）
    print(f'[dl={default_label}]', end='')
    valid = labels != default_label
    if not np.any(valid):
        return np.nan
    positions = positions[valid]
    labels = labels[valid]
    unique_classes = np.unique(labels)
    n_classes = len(unique_classes)

    if n_classes < 2:
        return np.nan

    # 1. 计算每个簇的【质心】和【簇内平均半径】
    centroids = []
    radii = []

    for cls in unique_classes:
        # 获取该簇的所有点
        cls_points = positions[labels == cls]

        # 计算质心 (Centroid)
        centroid = np.mean(cls_points, axis=0)
        centroids.append(centroid)

        # 计算该簇内每个点到质心的距离
        # cdist 输入需要是 2D 数组, 所以将 centroid 扩充维度
        dists_to_center = cdist(cls_points, [centroid], metric='euclidean')

        # 计算平均半径 (Average Radius)
        avg_radius = np.mean(dists_to_center)
        radii.append(avg_radius)

    # 2. 计算 d_intra (所有簇平均半径的均值)
    d_intra = np.mean(radii)

    # 3. 计算 d_inter (所有簇质心之间的平均距离)
    centroids = np.array(centroids)

    # pdist 计算两两质心之间的距离
    centroid_dists = pdist(centroids, metric='euclidean')

    if centroid_dists.size == 0:
        d_inter = 0
    else:
        d_inter = np.mean(centroid_dists)

    # 4. 计算 CS 指标：越大表示簇分离效果越好
    # CS = d_inter / (d_intra + d_inter)
    if (d_intra + d_inter) == 0:
        cs_score = 0.0  # 无法区分
    else:
        cs_score = d_inter / (d_intra + d_inter)

    return cs_score
