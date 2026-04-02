#!/usr/bin/env python3
"""
将 LinLog 输出的 .lc/.txt 转为按节点编号排序的坐标文件。
输出格式：第 i 行为节点 i 的 x y(仅两列)，节点编号从 0 开始连续。
若某节点在输入中不存在，该行输出 0 0。

用法：python postprocess.py <input.txt> <output.txt>
"""

import sys

if len(sys.argv) != 3:
    print(f"用法: {sys.argv[0]} <input_file> <output_file>")
    sys.exit(1)

input_file = sys.argv[1]
output_file = sys.argv[2]

# 解析: 第一行为时间，之后每行 node_id x y z cluster
node_to_xy = {}
time_line = None
with open(input_file, "r") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        if time_line is None:
            time_line = line
            continue
        parts = line.split()
        if len(parts) < 4:
            continue
        nid = int(parts[0])
        if (nid == 119043):
            print(line)
        x, y = float(parts[1]), float(parts[2])
        node_to_xy[nid] = (x, y)

if not node_to_xy:
    print(f"No valid lines in input: {input_file}")
    sys.exit(1)

max_id = max(node_to_xy.keys())
print(f'max id : {max_id}')

with open(output_file, "w") as f:
    print(f"layout time: {time_line} s")
    for i in range(max_id + 1):
        if i in node_to_xy:
            x, y = node_to_xy[i]
            f.write(f"{x} {y}\n")
        # else:
        #     print(f"error: node {i} not found.")
