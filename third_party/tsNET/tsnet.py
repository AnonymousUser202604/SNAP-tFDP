#!/usr/bin/env python3

import os
import sys
import time
import argparse
import numpy as np
import graph_tool.all as gt
import modules.distance_matrix as distance_matrix
import modules.thesne as thesne


def load_mtx_graph(path):
    """从 mtx 边表文件读图，返回 graph_tool 无向图，节点为 0..n-1。"""
    edges = []
    max_idx = -1
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('%'):
                continue
            parts = line.split()
            if len(parts) >= 2:
                i, j = int(parts[0]), int(parts[1])
                if i != j:
                    edges.append((i, j))
                max_idx = max(max_idx, i, j)

    n = max_idx + 1
    g = gt.Graph(directed=False)
    g.add_vertex(n)
    for i, j in edges:
        g.add_edge(i, j)

    gt.remove_parallel_edges(g)
    return g


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='tsNET graph layout algorithm')
    parser.add_argument('--input', required=True, help='Input edge file (MTX format)')
    parser.add_argument('--pmds_layout', required=True, help='Initial layout file (txt format)')
    parser.add_argument('--output', required=True, help='Output layout file')
    parser.add_argument('--perplexity', type=float, default=150, help='Perplexity parameter (default: 150)')
    parser.add_argument('--learning_rate', type=float, default=50, help='Learning rate (default: 50)')

    args = parser.parse_args()

    # Validate input files
    if not os.path.isfile(args.input):
        print(f'Error: graph file not found: {args.input}')
        sys.exit(1)
    if not os.path.isfile(args.pmds_layout):
        print(f'Error: init file not found: {args.pmds_layout}')
        sys.exit(1)

    # Global hyperparameters
    n_iters = 5000
    momentum = 0.5
    tolerance = 1e-7
    window_size = 40
    r_eps = 0.05
    lambdas_2 = [1, 1.2, 0]
    lambdas_3 = [1, 0.01, 0.6]
    output_dims = 2

    print('Reading graph:', args.input)
    g = load_mtx_graph(args.input)
    n = g.num_vertices()
    print('Graph: |V|={}, |E|={}'.format(n, g.num_edges()))

    print('Computing shortest-path distance matrix...')
    X = distance_matrix.get_distance_matrix(g, 'spdm', verbose=False)
    print('Done.')

    print('Loading init layout:', args.pmds_layout)
    Y_init = np.loadtxt(args.pmds_layout, dtype=np.float64)
    assert Y_init.shape[0] == n and Y_init.shape[1] == 2

    start_time = time.time()

    # The actual optimization is done in the thesne module.
    Y = thesne.tsnet(
        X, output_dims=output_dims, random_state=1, perplexity=args.perplexity, n_epochs=n_iters,
        Y=Y_init,
        initial_lr=args.learning_rate, final_lr=args.learning_rate, lr_switch=n_iters // 2,
        initial_momentum=momentum, final_momentum=momentum, momentum_switch=n_iters // 2,
        initial_l_kl=lambdas_2[0], final_l_kl=lambdas_3[0], l_kl_switch=n_iters // 2,
        initial_l_c=lambdas_2[1], final_l_c=lambdas_3[1], l_c_switch=n_iters // 2,
        initial_l_r=lambdas_2[2], final_l_r=lambdas_3[2], l_r_switch=n_iters // 2,
        r_eps=r_eps, autostop=tolerance, window_size=window_size,
        verbose=True
    )

    # Normalize layout to [0,1]
    Y_cpy = Y.copy()
    for dim in range(Y.shape[1]):
        Y_cpy[:, dim] += -Y_cpy[:, dim].min()
    scaling = 1 / (np.absolute(Y_cpy).max())
    Y = Y_cpy * scaling

    elapsed = time.time() - start_time
    print('tsNET running time: {:.6f} s'.format(elapsed))

    # Create output directory and save
    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    with open(args.output, 'w') as f:
        np.savetxt(f, Y, delimiter=' ', fmt='%.6f')
    print('Saved:', args.output)
