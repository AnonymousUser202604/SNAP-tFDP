#!/usr/bin/env python3
from tfdp import tFDP
import numpy as np
import os
import sys
import argparse


def main():
    parser = argparse.ArgumentParser(description='t-FDP graph layout algorithm')
    parser.add_argument('input_edges', help='Input edge file (MTX format)')
    parser.add_argument('pmds_init', help='Initial layout file (PMDS)')
    parser.add_argument('output', help='Output layout file')
    parser.add_argument('--algo', default='ibFFT_CPU', help='Algorithm to use (default: ibFFT_CPU)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed (default: 42)')
    parser.add_argument('--n-threads', type=int, default=8, help='Number of threads for parallel algorithms (default: 8)')

    args = parser.parse_args()

    # 验证输入文件
    if not os.path.isfile(args.input_edges):
        print(f'Error: graph file not found: {args.input_edges}')
        sys.exit(1)
    if not os.path.isfile(args.pmds_init):
        print(f'Error: init file not found: {args.pmds_init}')
        sys.exit(1)

    # 获取文件名（不含扩展名）用于日志
    filename = os.path.splitext(os.path.basename(args.input_edges))[0]

    print(f'Loading initial layout: {args.pmds_init}')
    posinit = np.loadtxt(args.pmds_init, dtype=np.float64)

    print(f'Initializing t-FDP with algorithm: {args.algo}')
    tfdp = tFDP(init=posinit, algo=args.algo, randseed=args.seed, filename=filename, n_threads=args.n_threads)

    print(f'Reading graph: {args.input_edges}')
    tfdp.readgraph(args.input_edges)
    print(f"Graph: {filename}")
    tfdp.graphinfo()

    print('Running optimization...')
    res, t = tfdp.optimization()

    print(f"\nOptimization time: {t:.3f} seconds")

    # 创建输出目录
    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)

    print(f'Saving result to: {args.output}')
    with open(args.output, 'w') as f:
        np.savetxt(f, res, fmt="%.6f", delimiter=" ")

    print('Done!')


if __name__ == '__main__':
    main()