import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score


def calculate_cq_ari(node_positions, ground_truth_labels, n_clusters=None, default_label=-1):
    """
    计算 CQ_ARI 指标
    :param node_positions: 形状为 (n_nodes, 2) 的数组 表示图布局后的节点坐标
    :param ground_truth_labels: 每个节点真实的聚类标签列表(缺省社区标签由 default_label 指定,  会被跳过)
    :param n_clusters: 聚类数量。如果为 None 则默认使用真实标签的类别数量(不含缺省)
    :param default_label: 缺省社区标签,  该标签的节点不纳入计算。ACO/APH/co_author_8391 为 -1,  其余为 0
    :return: CQ_ARI 分数
    """
    # 跳过缺省标签的节点（缺省社区不纳入指标计算）
    print(f'[dl={default_label}]', end='')
    valid = (np.asarray(ground_truth_labels) != default_label)
    if not np.any(valid):
        return np.nan  # 无有效节点
    positions_valid = np.asarray(node_positions)[valid]
    labels_valid = np.asarray(ground_truth_labels)[valid]

    # 1. 确定聚类数量 (k)
    if n_clusters is None:
        n_clusters = len(np.unique(labels_valid))
    if n_clusters <= 0 or len(positions_valid) < n_clusters:
        return np.nan

    # 2. 对节点坐标进行几何聚类 (Geometric Clustering)
    # 论文中提到在绘图中忽略边,  只根据坐标进行聚类
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
    geometric_labels = kmeans.fit_predict(positions_valid)

    # 3. 计算调整兰德系数 (ARI)
    # ARI 的范围是 [-1, 1],  1 表示完美匹配,  0 或负数表示随机匹配
    cq_ari_score = adjusted_rand_score(labels_valid, geometric_labels)

    return cq_ari_score
