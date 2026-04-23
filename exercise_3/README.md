# Exercise 3 – Forest Fire Cellular Automaton (NASA FIRMS)

## Objective
Simulate forest fire propagation using a 2-D cellular automaton initialized
with real hotspot detections from NASA FIRMS, and evaluate the performance of
a serial vs MPI-parallel implementation.

## Files
| File | Description |
|---|---|
| `serial_fire.py` | Serial baseline: fetches FIRMS data, builds grid, runs automaton, plots results |
| `parallel_fire_mpi.py` | Parallel version using `mpi4py` with domain decomposition (horizontal slabs) |
| `benchmark.py` | Runs both versions at multiple grid sizes and generates comparison plots |

## States
| Code | Meaning |
|---|---|
| 0 | Non-burnable / outside domain |
| 1 | Susceptible vegetation |
| 2 | Burning |
| 3 | Burned |

## Requirements
```
pip install numpy pandas matplotlib requests mpi4py
```

## Running

### Serial
```bash
python serial_fire.py
```
Outputs go to `outputs/`:
- `hotspots.csv` – FIRMS detections used
- `serial_snapshots.png` – grid snapshots at selected steps
- `serial_stats.png` – time-series of cell-state counts
- `serial_stats.csv` – raw per-step counts
- `serial_timing.json` – timing record

### Parallel (MPI)
```bash
mpiexec -n 4 python parallel_fire_mpi.py --rows 200 --cols 200 --steps 100
```
Outputs:
- `outputs/parallel_stats.csv`
- `outputs/parallel_timing.json`

### Benchmark (serial + MPI comparison)
```bash
python benchmark.py
```
Outputs:
- `outputs/benchmark_results.csv`
- `outputs/benchmark_plot.png`

## NASA FIRMS API key
Set the `FIRMS_API_KEY` environment variable or replace the placeholder in
`serial_fire.py`.  Without a key the scripts generate **synthetic** hotspot
data that replicates the same spatial format.

```bash
export FIRMS_API_KEY="your_key_here"
python serial_fire.py
```

## Model parameters
| Parameter | Default | Description |
|---|---|---|
| `GRID_ROWS / GRID_COLS` | 200 | Grid resolution |
| `N_STEPS` | 100 | Simulation steps |
| `BURN_LIFETIME` | 3 | Steps a cell burns before becoming ash |
| `SPREAD_PROB_BASE` | 0.4 | Base spread probability |
| `FRP_SCALE` | 0.01 | FRP influence on spread probability |
