# SNAP-tFDP: Massively Scalable Graph Layouts via  Sparse Negative Sampling

SNAP-tFDP (Stochastic Negative-sampling Accelerated Placement t-FDP) achieves $O(|E|)$ time complexity with a low memory footprint, without requiring complex multi-level representations.

## Directory Structure

- `data/`: Dataset directory (large datasets need to be downloaded from external [link]()).
    - `*.txt`: Edge data files
    - `*.attr`: Node label files
- `include/`: Header files
- `src/`: Source code
    - `graph/`: Graph definitions
    - `layout/`: Layout algorithms
    - `metrics/`: Our custom multi-threaded NP metric implementation
- `figures/`: Figures used in the paper (SVG format)
- `scripts/`: Experiment scripts
- `results/`: Experiment outputs (2D embedding coordinates and images)
- `statistics/`: Metric statistics for experiment results
- `tools/`: Utility tools
- `third_party/`: Code repositories of baseline/comparison methods

---

# SNAP-tFDP Usage

## Build

```shell
mkdir build && cd build
cmake .. -DENABLE_PARALLEL=OFF -DENABLE_CUDA=OFF
cmake --build . --target snap-tfdp
```

If you want to enable the CPU parallel version (requires OpenMP), set `-DENABLE_PARALLEL=ON`.
If you want to enable the GPU parallel version (requires an NVIDIA GPU and CUDA Toolkit), set `-DENABLE_CUDA=ON`.

## Run

```shell
SNAP-tFDP

./snap-tfdp [OPTIONS] dataset output

POSITIONALS:
  dataset TEXT REQUIRED       Dataset path
  output TEXT REQUIRED        Output path

OPTIONS:
  -h,     --help              Print this help message and exit
          --init TEXT:{pmds,random,spiral}
                              Init method: pmds, random, spiral
          --pmds-file TEXT    PMDS initialization file path (required when --init pmds)
  -t,     --n-epoch INT:NONNEGATIVE
                              Number of epochs
  -k,     --k INT:POSITIVE    Negative sampling number (param k)
          --seed INT          Random seed
  -p,     --parallel          Enable CPU parallel mode
  -g,     --gpu               Enable GPU parallel mode
          --n-threads INT     Number of threads (default: -1 for max, effective with --parallel
                              or --gpu)
```
