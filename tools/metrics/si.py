import numpy as np
import cupy as cp
from cuml.metrics.cluster import silhouette_score


# from sklearn.metrics import silhouette_score


def calculate_silhouette(positions, labels, default_label=-1):
    """
    计算 Silhouette Score (轮廓系数)

    参数:
    positions: (N, 2) 节点坐标矩阵
    labels: (N,) 节点标签数组
    default_label: 缺省社区标签, 不纳入计算。ACO/APH/co_author_8391 为 -1, 其余为 0

    返回:
    float: Silhouette Score 值，范围 [-1, 1]，越大表示聚类质量越好
    """
    positions = np.asarray(positions)
    labels = np.asarray(labels)

    # 跳过缺省标签的节点（缺省社区不纳入计算）
    print(f'[dl={default_label}]', end='')
    valid = labels != default_label
    if not np.any(valid):
        return np.nan

    positions = positions[valid]
    labels = labels[valid]

    # 若超过100k个点，随机均匀采样100k个
    n_samples = len(labels)
    sample_limit = 400000
    if n_samples > sample_limit:
        sample_indices = np.random.choice(n_samples, size=sample_limit, replace=False)
        positions = positions[sample_indices]
        labels = labels[sample_indices]
        print(f'[sampled {sample_limit}/{n_samples}]', end='')

    unique_classes = np.unique(labels)
    n_classes = len(unique_classes)

    # Silhouette Score 至少需要 2 个类别
    if n_classes < 2:
        return np.nan

    positions_gpu = cp.asarray(positions, dtype=cp.float32)
    labels_gpu = cp.asarray(labels, dtype=cp.int32)
    print(labels_gpu.shape)
    score = silhouette_score(positions_gpu, labels_gpu, metric='euclidean')

    # score = silhouette_score(positions, labels, metric='euclidean')

    # 转回 Python float
    return float(score)
