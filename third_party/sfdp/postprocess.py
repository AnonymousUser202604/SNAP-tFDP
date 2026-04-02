#!/usr/bin/env python3
import sys
import argparse
import numpy as np


def main():
    parser = argparse.ArgumentParser(description='Post-process sfdp layout output')
    parser.add_argument('input', help='Input file from sfdp (id x y format)')
    parser.add_argument('output', help='Output file (x y format, sorted by id)')

    args = parser.parse_args()

    # 读取 sfdp 输出
    data = []
    with open(args.input, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) >= 3:
                try:
                    node_id = int(parts[0])
                    x = float(parts[1])
                    y = float(parts[2])
                    data.append((node_id, x, y))
                except (ValueError, IndexError):
                    continue

    if not data:
        print(f'Error: No valid data found in {args.input}')
        sys.exit(1)

    # 按 id 排序
    data.sort(key=lambda item: item[0])

    # 提取坐标
    coords = np.array([[x, y] for _, x, y in data], dtype=np.float64)

    # 保存输出
    with open(args.output, 'w') as f:
        np.savetxt(f, coords, fmt='%.6f', delimiter=' ')

    print(f'Saved {len(coords)} nodes to {args.output}')


if __name__ == '__main__':
    main()
