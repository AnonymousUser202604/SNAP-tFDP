#!/usr/bin/env python3
"""
Parameter iteration metrics plotter with dual-axis support
Plots SI (left y-axis) and QGG (right y-axis) metrics
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import argparse
from pathlib import Path


def plot_param_iter(methods, base_dir, datasets=None, output_dir=None, iters=None):
    """
    Plot SI curves for specified methods and datasets

    Args:
        methods: List of method names (e.g., ['nsgl_par', 'nsgl', 'nsgl_gpu'])
        base_dir: Base directory containing CSV files
        datasets: List of dataset names (if None, auto-detect from files)
        output_dir: Output directory for plots (if None, use base_dir)
        iters: List of iteration values to visualize (if None, use all)
    """

    base_dir = Path(base_dir)
    if output_dir is None:
        output_dir = base_dir
    else:
        output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Auto-detect datasets if not provided
    if datasets is None:
        datasets = set()
        for file in base_dir.glob("*.si.csv"):
            # Extract dataset name from filename like "APH.nsgl_par.si.csv"
            parts = file.stem.split('.')
            if len(parts) >= 2:
                dataset = parts[0]
                datasets.add(dataset)
        datasets = sorted(list(datasets))

    if not datasets:
        print(f"Error: No datasets found in {base_dir}")
        return

    print(f"Found datasets: {datasets}")
    print(f"Methods: {methods}")

    # Color and marker mapping for methods
    colors = {
        'nsgl': 'blue',
        'nsgl_par': 'orange',
        'nsgl_gpu_8192': 'green',
    }

    markers = {
        'nsgl': 's',
        'nsgl_par': 'o',
        'nsgl_gpu_8192': '^',
    }

    marker_size = 80

    line_width = 3

    labels = {
        'nsgl': 'SNAP-tFDP-Serial',
        'nsgl_par': 'SNAP-tFDP-Parallel(CPU)',
        'nsgl_gpu_8192': 'SNAP-tFDP-Parallel(GPU)',
    }

    # Plot for each dataset
    for dataset in datasets:
        scaling = 1.15
        fig, ax = plt.subplots(figsize=(12 * scaling, 4.5 * scaling))

        has_data = False

        # Set labels and ticks
        ax.set_xlabel('Number of epochs', fontsize=32, color='black')
        ax.set_ylabel('SI', fontsize=32, color='black')
        ax.tick_params(axis='y', labelsize=28, labelcolor='black')
        ax.tick_params(axis='x', labelsize=28)

        # Set x-axis to show ticks at 10, 20, 30, etc.
        ax.xaxis.set_major_locator(plt.MultipleLocator(10))

        for method in methods:
            # Load SI data
            si_csv_file = base_dir / f"{dataset}.{method}.si.csv"

            si_data = None

            if si_csv_file.exists():
                try:
                    si_data = pd.read_csv(si_csv_file)
                    if si_data.empty:
                        print(f"Warning: {si_csv_file} is empty")
                        si_data = None
                except Exception as e:
                    print(f"Error reading {si_csv_file}: {e}")

            if si_data is None:
                print(f"Warning: No data found for method {method}, skipping...")
                continue

            has_data = True

            # Get color and marker for this method
            color = colors.get(method, 'gray')
            marker = markers.get(method, 'o')
            label = labels.get(method, method)

            # Process SI data
            if 'iter' in si_data.columns:
                si_iters = si_data['iter'].values
                si_values = si_data['si'].values
            else:
                si_iters = np.arange(len(si_data))
                si_values = si_data['si'].values

            # Filter by iters if specified
            if iters is not None:
                mask = np.isin(si_iters, iters)
                si_iters = si_iters[mask]
                si_values = si_values[mask]

            # Plot SI scatter points
            ax.scatter(si_iters, si_values, marker=marker, color=color, s=marker_size,
                       alpha=0.5, edgecolors=color, linewidths=1)

            # Calculate and plot SI line
            unique_iters = sorted(set(si_iters))
            avg_si = []
            for iter_val in unique_iters:
                mask = si_iters == iter_val
                si_vals = si_values[mask]
                avg_si.append(np.mean(si_vals))

            ax.plot(unique_iters, avg_si, color=color, linewidth=line_width,
                    linestyle='-', marker=marker, markersize=6, markeredgewidth=1,
                    label=f'{label}', alpha=0.8)

            print(f"✓ Loaded SI: {si_csv_file}")

        if not has_data:
            print(f"Warning: No valid data for dataset {dataset}, skipping...")
            plt.close()
            continue

        # Add grid
        ax.grid(True, alpha=0.3, linestyle='--')

        # Create legend at bottom right with line and marker
        ax.legend(fontsize=28, loc='lower right', framealpha=0.95, handlelength=2.5)

        # Adjust layout
        plt.tight_layout()

        # Save figure
        output_file = output_dir / f"{dataset}_si_comparison.svg"
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"✓ Saved: {output_file}")
        plt.close()

    print("\nAll plots generated successfully!")


def main():
    parser = argparse.ArgumentParser(
        description='Plot SI curves for graph layout methods',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Plot all three methods
  python3 plot_param_iter.py --methods nsgl_par nsgl nsgl_gpu_8192 --base-dir ./statistics/param_iter/metrics

  # Plot only parallel vs GPU
  python3 plot_param_iter.py --methods nsgl_par nsgl_gpu_8192 --base-dir ./statistics/param_iter/metrics

  # Specify output directory
  python3 plot_param_iter.py --methods nsgl_par nsgl --base-dir ./statistics/param_iter/metrics --output-dir ./figures

  # Specify datasets explicitly
  python3 plot_param_iter.py --methods nsgl_par nsgl --base-dir ./statistics/param_iter/metrics --datasets APH socfb-UF21 com-lj

  # Specify iterations to visualize
  python3 plot_param_iter.py --methods nsgl_par nsgl --base-dir ./statistics/param_iter/metrics --iters 1 2 5 10 20 50 100
        """
    )

    parser.add_argument('--methods', nargs='+', default=['nsgl_par', 'nsgl', 'nsgl_gpu_8192'],
                        help='Methods to plot (default: nsgl_par nsgl nsgl_gpu_8192)')
    parser.add_argument('--base-dir', default='./statistics/param_iter/metrics',
                        help='Base directory containing CSV files (default: ./statistics/param_iter/metrics)')
    parser.add_argument('--output-dir', default=None,
                        help='Output directory for plots (default: same as base-dir)')
    parser.add_argument('--datasets', nargs='+', default=None,
                        help='Dataset names to plot (default: auto-detect from files)')
    parser.add_argument('--iters', nargs='+', type=int, default=None,
                        help='Iteration values to visualize (default: all iterations)')

    args = parser.parse_args()

    plot_param_iter(
        methods=args.methods,
        base_dir=args.base_dir,
        datasets=args.datasets,
        output_dir=args.output_dir,
        iters=args.iters
    )


if __name__ == '__main__':
    main()
