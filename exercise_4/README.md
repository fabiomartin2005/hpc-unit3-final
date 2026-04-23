# Exercise 4 – Parallel K-Means Clustering (Covertype Dataset)

## Objective
Implement K-Means on the UCI Covertype dataset in serial and parallel (MPI)
versions, study how data partitioning and collective communication enable
scaling, and compare runtimes, speedup, and clustering quality.

## Files
| File | Description |
|---|---|
| `serial_kmeans.py` | Serial K-Means with K-Means++ initialisation, PCA visualisation |
| `parallel_kmeans_mpi.py` | Parallel K-Means using `mpi4py` (row partitioning + Allreduce) |
| `benchmark.py` | Runs both versions at multiple k values and process counts |

## Parallelisation Strategy
Each MPI process owns a contiguous shard of the dataset rows.

**Per iteration:**
1. Each process assigns its local points to the nearest centroid.
2. Each process accumulates local sums and counts per cluster.
3. `MPI.Allreduce(SUM)` aggregates across all processes → global centroids.
4. Convergence is checked collectively with `allreduce(MIN)`.

This avoids any per-sample communication; only `k × d` floating-point values
are communicated per iteration.

## Requirements
```
pip install numpy pandas matplotlib scikit-learn mpi4py
```

## Running

### Serial
```bash
python serial_kmeans.py --k 7 --max_rows 100000
```
Outputs in `outputs/`:
- `serial_timing.json`
- `serial_iter_times.csv`
- `serial_labels.npy`, `serial_centers.npy`
- `serial_pca.png` – 2-D PCA cluster visualisation

### Parallel (MPI)
```bash
mpiexec -n 4 python parallel_kmeans_mpi.py --k 7 --max_rows 100000
```
Outputs:
- `outputs/parallel_timing.json`
- `outputs/parallel_iter_times.csv`
- `outputs/parallel_labels.npy`, `outputs/parallel_centers.npy`

### Benchmark
```bash
python benchmark.py
```
Outputs:
- `outputs/benchmark_results.csv`
- `outputs/benchmark_plot.png`
- `outputs/iter_time_comparison.png`

## Dataset
The [Covertype dataset](https://archive.ics.uci.edu/ml/datasets/covertype) is
downloaded automatically to `data/covtype.csv` on the first run.

- **Samples:** 581 012
- **Features:** 54 (10 continuous + 44 binary)
- **Target:** 7 forest cover types (used as ground-truth for evaluation only)

## Parameters
| Parameter | Default | Description |
|---|---|---|
| `--k` | 7 | Number of clusters |
| `--max_iter` | 50 | Maximum iterations |
| `--max_rows` | 100 000 | Sub-sample for fast testing |
| `TOL` | 1e-4 | Centroid shift convergence threshold |
