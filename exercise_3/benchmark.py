"""
Exercise 3 – Benchmark: serial vs parallel MPI for different grid sizes.

Usage (run from exercise_3/):
    python benchmark.py

This script:
  1. Runs the serial implementation at multiple grid sizes.
  2. Calls mpiexec for 2 and 4 processes at each size.
  3. Collects timing JSON files and produces a comparison plot.
"""

import subprocess
import json
import os
import sys
import pandas as pd
import matplotlib.pyplot as plt

GRID_SIZES  = [100, 150, 200]   # (rows = cols for each)
N_STEPS     = 50
MPI_PROCS   = [1, 2, 4]
OUT_DIR     = "outputs"


def run_serial(rows, cols, steps):
    import importlib.util, sys as _sys
    # import serial module directly
    spec = importlib.util.spec_from_file_location("sf", "serial_fire.py")
    sf   = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sf)

    hotspots = sf._synthetic_hotspots(sf.REGION)
    grid, frp_map, *_ = sf.build_full_grid(sf.REGION, rows, cols, hotspots) \
        if hasattr(sf, "build_full_grid") else sf.build_grid(sf.REGION, rows, cols, hotspots)
    _, _, _, elapsed = sf.run_serial(grid, frp_map, steps, save_snapshots=False)
    return elapsed


def run_mpi(rows, cols, steps, nprocs):
    result = subprocess.run(
        ["mpiexec", "-n", str(nprocs),
         sys.executable, "parallel_fire_mpi.py",
         "--rows", str(rows), "--cols", str(cols), "--steps", str(steps)],
        capture_output=True, text=True
    )
    # parse elapsed from output json written by rank 0
    timing_file = os.path.join(OUT_DIR, "parallel_timing.json")
    if os.path.exists(timing_file):
        with open(timing_file) as f:
            data = json.load(f)
        return data["elapsed_s"]
    # fallback: parse stdout
    for line in result.stdout.splitlines():
        if "Elapsed" in line:
            try:
                return float(line.split("Elapsed:")[1].split("s")[0].strip())
            except Exception:
                pass
    return None


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    records = []

    for size in GRID_SIZES:
        print(f"\n=== Grid {size}x{size}, steps={N_STEPS} ===")

        # serial
        t_serial = run_serial(size, size, N_STEPS)
        print(f"  serial:       {t_serial:.4f} s")
        records.append({"grid": f"{size}x{size}", "mode": "serial",
                         "procs": 1, "elapsed_s": t_serial})

        # MPI (2 and 4 processes)
        for np_ in [2, 4]:
            t_mpi = run_mpi(size, size, N_STEPS, np_)
            if t_mpi:
                speedup = t_serial / t_mpi
                print(f"  mpi (n={np_}):    {t_mpi:.4f} s  speedup={speedup:.2f}x")
                records.append({"grid": f"{size}x{size}", "mode": f"mpi_n{np_}",
                                 "procs": np_, "elapsed_s": t_mpi,
                                 "speedup": speedup})
            else:
                print(f"  mpi (n={np_}):    [failed]")

    df = pd.DataFrame(records)
    df.to_csv(os.path.join(OUT_DIR, "benchmark_results.csv"), index=False)

    # plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # elapsed time
    ax = axes[0]
    for mode, grp in df.groupby("mode"):
        ax.plot(grp["grid"], grp["elapsed_s"], marker="o", label=mode)
    ax.set_title("Elapsed time vs grid size")
    ax.set_xlabel("Grid size")
    ax.set_ylabel("Seconds")
    ax.legend()

    # speedup
    ax = axes[1]
    speedup_df = df[df["mode"] != "serial"].dropna(subset=["speedup"])
    for mode, grp in speedup_df.groupby("mode"):
        ax.plot(grp["grid"], grp["speedup"], marker="s", label=mode)
    ax.axhline(1, color="grey", linestyle="--", linewidth=0.8)
    ax.set_title("Speedup vs grid size")
    ax.set_xlabel("Grid size")
    ax.set_ylabel("Speedup (serial / parallel)")
    ax.legend()

    plt.tight_layout()
    out = os.path.join(OUT_DIR, "benchmark_plot.png")
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"\n[INFO] Benchmark plot saved → {out}")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
