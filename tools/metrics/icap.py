import numpy as np
from scipy.spatial.distance import cdist
from scipy.stats import spearmanr


def calculate_icap(positions, labels, edges, default_label=-1):
    """
    计算基于秩相关性的 Inter-Cluster Affinity Preservation (ICAP)
    缺省社区标签由 default_label 指定, 不参与计算（不建簇、不统计与之相关的边）。
    ACO/APH/co_author_8391 的缺省标签为 -1, 其余为 0。

    公式逻辑:
    1. A_ij (亲和度) = 簇间边数 / (簇i大小 * 簇j大小)
    2. d_ij (距离) = 簇i质心与簇j质心的欧氏距离
    3. S_ij (相似度) = -d_ij
    4. 对每个簇 i 计算向量 A_i 和 S_i 的 Spearman 秩相关系数
    5. 对所有系数求和
    """
    labels = np.asarray(labels)
    # 跳过缺省标签, 只考虑非缺省社区
    print(f'[dl={default_label}]', end='')
    unique_classes = np.unique(labels)
    unique_classes = unique_classes[unique_classes != default_label]
    n_classes = len(unique_classes)

    # Spearman 相关系数至少需要 3 个数据点才能体现"排序"的意义 (对比自己以外的至少2个簇)
    if n_classes < 3:
        return np.nan

    # ---------------------------------------------------------
    # 第一步：准备基础数据 (簇大小、质心、映射表)
    # ---------------------------------------------------------
    cluster_sizes = {}
    centroids = []

    # 建立 label 到 0..k-1 索引的映射, 防止 label 是非连续整数
    label_to_idx = {lbl: i for i, lbl in enumerate(unique_classes)}

    # 计算质心和大小
    for cls in unique_classes:
        idx = np.where(labels == cls)[0]
        cluster_sizes[cls] = len(idx)
        # 计算质心 (Centroid)
        center = np.mean(positions[idx], axis=0)
        centroids.append(center)

    centroids = np.array(centroids)  # Shape: (k, 2)

    # ---------------------------------------------------------
    # 第二步：计算簇间连边矩阵 (Edge Count Matrix)
    # ---------------------------------------------------------
    # 初始化 k*k 矩阵
    adj_matrix = np.zeros((n_classes, n_classes), dtype=int)

    # 将edges转换为ndarray（如果不是的话）
    edges = np.asarray(edges)

    # 向量化处理：过滤有效边
    valid_edges = (edges[:, 0] < len(labels)) & (edges[:, 1] < len(labels))
    edges_valid = edges[valid_edges]

    # 获取边的标签
    labels_u = labels[edges_valid[:, 0]]
    labels_v = labels[edges_valid[:, 1]]

    # 过滤掉包含缺省标签的边和自环边
    mask = (labels_u != default_label) & (labels_v != default_label) & (labels_u != labels_v)
    edges_filtered = edges_valid[mask]
    labels_u_filtered = labels_u[mask]
    labels_v_filtered = labels_v[mask]

    # 映射到索引
    for i in range(len(edges_filtered)):
        l_u = labels_u_filtered[i]
        l_v = labels_v_filtered[i]
        idx_u = label_to_idx[l_u]
        idx_v = label_to_idx[l_v]
        # 无向图, 双向增加
        adj_matrix[idx_u, idx_v] += 1
        adj_matrix[idx_v, idx_u] += 1

    # ---------------------------------------------------------
    # 第三步：计算 Aij (Affinity) 和 Sij (Layout Similarity)
    # ---------------------------------------------------------

    # 3.1 计算 Affinity Matrix A
    # A_ij = m_ij / (|Ci| * |Cj|)
    affinity_matrix = np.zeros((n_classes, n_classes))

    for i in range(n_classes):
        for j in range(n_classes):
            if i != j:
                c_i = unique_classes[i]
                c_j = unique_classes[j]
                size_denom = cluster_sizes[c_i] * cluster_sizes[c_j]

                if size_denom > 0:
                    # affinity_matrix[i, j] = adj_matrix[i, j] / size_denom
                    affinity_matrix[i, j] = adj_matrix[i, j]
                else:
                    affinity_matrix[i, j] = 0

    # 3.2 计算 Distance Matrix D (质心距离)
    dist_matrix = cdist(centroids, centroids, metric='euclidean')

    # 3.3 计算 Similarity Matrix S = -D
    similarity_matrix = -dist_matrix

    # ---------------------------------------------------------
    # 第四步：计算秩相关系数并求和
    # ---------------------------------------------------------
    total_correlation = 0.0
    valid_clusters = 0

    for i in range(n_classes):
        # 提取当前簇 i 到其他所有簇 j 的向量
        # 需要排除 i 自身 (因为对自己距离为0, 亲和度无定义或最大, 会干扰排序)
        mask = np.arange(n_classes) != i

        vec_affinity = affinity_matrix[i, mask]
        vec_similarity = similarity_matrix[i, mask]

        # 检查向量是否有效（例如全为0则无法计算相关性）
        # 如果标准差为0, spearmanr 会返回 nan
        if np.std(vec_affinity) == 0 or np.std(vec_similarity) == 0:
            rho = 0  # 无法计算相关性, 通常视为无相关
        else:
            rho, p_val = spearmanr(vec_affinity, vec_similarity)
            if np.isnan(rho):
                rho = 0

        total_correlation += rho
        valid_clusters += 1

    # 通常为了便于对比, 计算平均值
    icap_avg = total_correlation / valid_clusters if valid_clusters > 0 else 0

    return icap_avg
