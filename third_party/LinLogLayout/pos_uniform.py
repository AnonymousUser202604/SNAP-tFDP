#!/usr/bin/env python3
"""
将 LinLog 输出的 .lc/.txt 转为按节点编号排序的坐标文件。
输出格式：第 i 行为节点 i 的 x y(仅两列) ，节点编号从 0 开始连续。
若某节点在输入中不存在，该行输出 0 0。
"""

import os

input_dir = "/home/sdu/wyf/LinLogLayout/tmp_result/2026_2_9"
output_dir = "/home/sdu/wyf/graduate2026/result/2026_2_9_tmp/Linlog"

for filename in os.listdir(input_dir):
    input_file = os.path.join(input_dir, filename)  
    output_file = os.path.join(output_dir, filename)
    if not os.path.isfile(input_file):
        continue

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
            x, y = float(parts[1]), float(parts[2])
            node_to_xy[nid] = (x, y)

    if not node_to_xy:
        print(f"No valid lines in input: {input_file}")
        continue

    max_id = max(node_to_xy.keys())

    with open(output_file, "w") as f:
        if time_line is not None:
            f.write(time_line + "\n")
        for i in range(max_id + 1):
            if i in node_to_xy:
                x, y = node_to_xy[i]
                f.write(f"{x} {y}\n")
            else:
                print(f"error: node {i} not found.")


# java -cp bin LinLogLayout 2 /home/sdu/wyf/graduate2026/data/co_author_8391.mtx tmp_result/co_author_8391.txt
