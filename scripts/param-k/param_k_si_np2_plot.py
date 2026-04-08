import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 确保图表目录存在
os.makedirs('./figures/param-k', exist_ok=True)

# 归一化开关
NORMALIZE = False  # True: 归一化, False: 原始值

# 读取数据
si_df = pd.read_csv('./statistics/param-k/si.result.csv')
np2_df = pd.read_csv('./statistics/param-k/np2.result.csv')
ari_df = pd.read_csv('./statistics/param-k/ari.result.csv')
time_df = pd.read_csv('./statistics/param-k/time.csv')
mem_df = pd.read_csv('./statistics/param-k/mem.csv')

# 选择的数据集
datasets = [
    "APH",
    "aircraft",
    "co_author_8391",
    "socfb-Yale4",
    "ACO",
    "socfb-UF21",
    "soc-Flickr-ASU",
    "com-dblp",
    "com-amazon",
    "com-youtube",
    "com-orkut",
    "com-lj",
]

# 提取k值（从列名中提取数字）
k_columns = [col for col in si_df.columns if col.startswith('k_')]
k_values = [int(col.split('_')[1]) for col in k_columns]

# 绘制 SI 指标
fig1, ax1 = plt.subplots(figsize=(7, 5))

# 存储所有值用于计算均值
all_si = []

# 绘制所有数据集的灰色曲线
for dataset in datasets:
    row = si_df[si_df['dataset'] == dataset]
    if not row.empty:
        values = row[k_columns].values.flatten()
        if NORMALIZE:
            # SI归一化：先加1变成[0,2]，然后除以最大值
            values_shifted = values + 1
            values = values_shifted / values_shifted.max()
        all_si.append(values)
        ax1.plot(k_values, values, marker='o', color='gray', linewidth=1.5, markersize=4, alpha=0.5)

# 计算并绘制均值曲线
mean_si = np.mean(all_si, axis=0)
ax1.plot(k_values, mean_si, marker='o', color='red', linewidth=2.5, markersize=6, label='Mean', zorder=10)

ax1.set_xlabel('k', fontsize=18)
ax1.set_ylabel('Normalized SI' if NORMALIZE else 'SI', fontsize=18)
ax1.legend(loc='upper right', fontsize=18)
ax1.set_xticks(range(1, 11))
ax1.tick_params(axis='both', labelsize=14)
ax1.grid(True, alpha=0.3)

# 设置纵轴范围，留出5%的上边距
y_min, y_max = ax1.get_ylim()
y_range = y_max - y_min
ax1.set_ylim(y_min, y_max + y_range * 0.05)
plt.tight_layout()
plt.savefig('./figures/param-k/param_k_si.svg', bbox_inches='tight')
print("SI图表已保存")
plt.close()

# 绘制 NP2 指标
fig2, ax2 = plt.subplots(figsize=(7, 5))

# 存储所有值用于计算均值
all_np2 = []

# 绘制所有数据集的灰色曲线
for dataset in datasets:
    row = np2_df[np2_df['dataset'] == dataset]
    if not row.empty:
        values = row[k_columns].values.flatten()
        if NORMALIZE:
            # NP2归一化：直接除以最大值
            values = values / values.max()
        all_np2.append(values)
        ax2.plot(k_values, values, marker='o', color='gray', linewidth=1.5, markersize=4, alpha=0.5)

# 计算并绘制均值曲线
mean_np2 = np.mean(all_np2, axis=0)
ax2.plot(k_values, mean_np2, marker='o', color='red', linewidth=2.5, markersize=6, label='Mean', zorder=10)

ax2.set_xlabel('k', fontsize=18)
ax2.set_ylabel('Normalized NP' if NORMALIZE else 'NP', fontsize=18)
ax2.legend(loc='upper right', fontsize=18)
ax2.set_xticks(range(1, 11))
ax2.tick_params(axis='both', labelsize=14)
ax2.grid(True, alpha=0.3)

# 设置纵轴范围，留出5%的上边距
y_min, y_max = ax2.get_ylim()
y_range = y_max - y_min
ax2.set_ylim(y_min, y_max + y_range * 0.05)
plt.tight_layout()
plt.savefig('./figures/param-k/param_k_np2.svg', bbox_inches='tight')
print("NP2图表已保存")
plt.close()

# 绘制 ARI 指标
fig3, ax3 = plt.subplots(figsize=(7, 5))

# 存储所有值用于计算均值
all_ari = []

# 绘制所有数据集的灰色曲线
for dataset in datasets:
    row = ari_df[ari_df['dataset'] == dataset]
    if not row.empty:
        values = row[k_columns].values.flatten()
        if NORMALIZE:
            # ARI归一化：先加1变成[0,2]，然后除以最大值
            values_shifted = values + 1
            values = values_shifted / values_shifted.max()
        all_ari.append(values)
        ax3.plot(k_values, values, marker='o', color='gray', linewidth=1.5, markersize=4, alpha=0.5)

# 计算并绘制均值曲线
mean_ari = np.mean(all_ari, axis=0)
ax3.plot(k_values, mean_ari, marker='o', color='red', linewidth=2.5, markersize=6, label='Mean', zorder=10)

ax3.set_xlabel('k', fontsize=18)
ax3.set_ylabel('Normalized CQ' if NORMALIZE else 'CQ', fontsize=18)
ax3.legend(fontsize=18, loc='upper right')
ax3.set_xticks(range(1, 11))
ax3.tick_params(axis='both', labelsize=14)
ax3.grid(True, alpha=0.3)

# 设置纵轴范围，留出5%的上边距
y_min, y_max = ax3.get_ylim()
y_range = y_max - y_min
ax3.set_ylim(y_min, y_max + y_range * 0.05)
plt.tight_layout()
plt.savefig('./figures/param-k/param_k_cq.svg', bbox_inches='tight')
print("CQ图表已保存")
plt.close()

# 绘制 Time 指标
fig4, ax4 = plt.subplots(figsize=(7, 5))

# 存储所有值用于计算均值
all_time = []

# 绘制所有数据集的灰色曲线
for dataset in datasets:
    row = time_df[time_df['dataset'] == dataset]
    if not row.empty:
        values = row[k_columns].values.flatten()
        if NORMALIZE:
            # Time归一化：直接除以最大值
            values = values / values.max()
        all_time.append(values)
        ax4.plot(k_values, values, marker='o', color='gray', linewidth=1.5, markersize=4, alpha=0.5)

# 计算并绘制均值曲线
mean_time = np.mean(all_time, axis=0)
ax4.plot(k_values, mean_time, marker='o', color='red', linewidth=2.5, markersize=6, label='Mean', zorder=10)

ax4.set_xlabel('k', fontsize=18)
ax4.set_ylabel('Normalized Time (s)' if NORMALIZE else 'Time (s)', fontsize=18)
# ax4.set_yscale('log')
ax4.legend(loc='upper right', fontsize=18)
ax4.set_xticks(range(1, 11))
ax4.tick_params(axis='both', labelsize=14)
ax4.grid(True, alpha=0.3)

# 设置纵轴范围，留出5%的上边距
y_min, y_max = ax4.get_ylim()
y_range = y_max - y_min
ax4.set_ylim(y_min, y_max + y_range * 0.05)
plt.tight_layout()
plt.savefig('./figures/param-k/param_k_time.svg', bbox_inches='tight')
print("Time图表已保存")
plt.close()

# 绘制 Mem 指标
fig5, ax5 = plt.subplots(figsize=(7, 5))

# 存储所有值用于计算均值
all_mem = []

# 绘制所有数据集的灰色曲线
for dataset in datasets:
    row = mem_df[mem_df['dataset'] == dataset]
    if not row.empty:
        values = row[k_columns].values.flatten()
        if NORMALIZE:
            # Mem归一化：直接除以最大值
            values = values / values.max()
        all_mem.append(values)
        ax5.plot(k_values, values, marker='o', color='gray', linewidth=1.5, markersize=4, alpha=0.5)

# 计算并绘制均值曲线
mean_mem = np.mean(all_mem, axis=0)
ax5.plot(k_values, mean_mem, marker='o', color='red', linewidth=2.5, markersize=6, label='Mean', zorder=10)

ax5.set_xlabel('k', fontsize=18)
ax5.set_ylabel('Normalized Mem (MB)' if NORMALIZE else 'Memory (MB)', fontsize=18)
ax5.set_yscale('log')
ax5.legend(loc='upper right', fontsize=18)
ax5.set_xticks(range(1, 11))
ax5.tick_params(axis='both', labelsize=14)
ax5.grid(True, alpha=0.3)

# 设置纵轴范围，留出5%的上边距
y_min, y_max = ax5.get_ylim()
y_range = y_max - y_min
ax5.set_ylim(y_min, y_max + y_range * 0.05)
plt.tight_layout()
plt.savefig('./figures/param-k/param_k_mem.svg', bbox_inches='tight')
print("Mem图表已保存")
plt.close()
