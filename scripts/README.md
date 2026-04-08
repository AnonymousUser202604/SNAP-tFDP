## Download large datasets(Optional)

Check [Osfstorage link](https://osf.io/dtxa5/files/osfstorage), download **all the three** files, put them into
`./data` folder and unzip `data.zip`:

```bash
unzip ./data/data.zip -d ./data
```

## Run SNAP-tFDP

```bash
bash ./scripts/run-snap-tfdp.sh
```

## Other scripts

- `benchmark/`: Run each comparison method, calculate metrics and plot.
- `case-study`: Layout the largest dataset `com-friendster` using SNAP-tFDP-Parallel
- `param-iter`: Run SNAP-tFDP with variable n_epoch
- `param-iter`: Run SNAP-tFDP with variable k
- `fig1`: Layout the dataset in figure1

## Tools

### `draw`

Draw a layout file to SVG.

#### Build

```bash
cmake -S . -B build
cmake --build build --target draw
# executable location: ./tools/draw
```

#### Usage

```bash
draw <input_layout.txt> <output.svg> <graph.txt|""|-> <labels.attr|""|->
```

use `-` to mean empty.

Examples:

```bash
# no edge, no color
draw layout.txt out.svg - -
```

```bash
# with edges, no color
draw layout.txt out.svg data/com-amazon.txt -
```

```bash
# with colors, no edges
draw layout.txt out.svg - data/com-amazon.attr
```

### `np`

Compute the NP metric from a graph file and a layout file.

#### Requirements

`np` requires OpenMP and [CGAL](https://github.com/CGAL/cgal).
On Ubuntu/Debian, install CGAL with:

```bash
sudo apt update
sudo apt install -y libcgal-dev
```

Then build:

```bash
# build metrics np
cmake -S . -B build -DMETRICS=ON
cmake --build build --target np
# executable location: ./tools/metrics/np
```

#### Usage

```bash
./tools/metrics/np <edges.txt> <pos.txt>
```

Example:

```bash
./tools/metrics/np data/com-amazon.txt results/com-amazon.txt
```

### `pmds` and `fr`

OGDF PivotMDS and Fruchterman-Reingold.  
The [OGDF code](https://github.com/ogdf/ogdf) is included in this repository as `third_party/OGDF`

#### Build

```bash
# build OGDF pmds and fr
cmake -S . -B build -DOGDF=ON
cmake --build build --target pmds --target fr
# executable location: ./pmds  ./fr
```

#### Usage

```bash
pmds <graph.txt> <result.txt>
```

```bash
fr <graph.txt> <pmds_init.txt> <result.txt>
```
