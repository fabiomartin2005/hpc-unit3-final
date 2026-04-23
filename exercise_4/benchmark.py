"""
Exercise 4 – Benchmark: serial vs MPI K-Means for different k and process counts.

Usage (from exercise_4/):
    python benchmark.py
"""

import subprocess
import json
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from serial_kmeans import load_covertype, kmeans_serial

K_VALUES    = [5, 7, 10]
MPI_PROCS   = [1, 2, 4]
MAX_ROWS    = 100_000
MAX_ITER    = 30
OUT_DIR     = "outputs"


def run_serial_timed(X, k):
    result = kmeans_serial(X, k=k, max_iter=MAX_ITER)
    return result["total_s"], result["inertia"], result["n_iter"]


def run_mpi_timed(k, nprocs):
    result = subprocess.run(
        ["mpiexec", "-n", str(nprocs),
         sys.executable, "parallel_kmeans_mpi.py",
         "--k", str(k), "--max_iter", str(MAX_ITER),
         "--max_rows", str(MAX_ROWS)],
        capture_output=True, text=True
    )
    timing_file = os.path.join(OUT_DIR, "parallel_timing.json")
    if os.path.exists(timing_file):
        with open(timing_file) as f:
            data = json.load(f)
        return data["total_s"], data.get("inertia", None), data["n_iter"]
    return None, None, None


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    print("[INFO] Loading dataset…")
    X, y, *_ = load_covertype(max_rows=MAX_ROWS)

    records = []

    for k in K_VALUES:
        print(f"\n=== k={k} ===")
        t_serial, inertia_s, niters = run_serial_timed(X, k)
        print(f"  serial:  {t_serial:.4f} s  (iter={niters})")
        records.append({"k": k, "procs": 1, "mode": "serial",
                         "total_s": t_serial, "inertia": inertia_s,
                         "speedup": 1.0})

        for np_ in MPI_PROCS[1:]:  # skip 1 (same as serial)
            t_mpi, inertia_p, niters_p = run_mpi_timed(k, np_)
            if t_mpi:
                speedup = t_serial / t_mpi
                print(f"  mpi n={np_}: {t_mpi:.4f} s  speedup={speedup:.2f}x  (iter={niters_p})")
                records.append({"k": k, "procs": np_, "mode": f"mpi_n{np_}",
                                 "total_s": t_mpi, "inertia": inertia_p,
                                 "speedup": speedup})
            else:
                print(f"  mpi n={np_}: [failed]")

    df = pd.DataFrame(records)
    df.to_csv(os.path.join(OUT_DIR, "benchmark_results.csv"), index=False)
    print("\n" + df.to_string(index=False))

    # ── plots ─────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # total time
    ax = axes[0]
    for mode, grp in df.groupby("mode"):
        ax.plot(grp["k"], grp["total_s"], marker="o", label=mode)
    ax.set_title("Total runtime vs k")
    ax.set_xlabel("k (clusters)")
    ax.set_ylabel("Seconds")
    ax.legend()

    # speedup
    ax = axes[1]
    speedup_df = df[df["procs"] > 1]
    for mode, grp in speedup_df.groupby("mode"):
        ax.plot(grp["k"], grp["speedup"], marker="s", label=mode)
    ax.axhline(1, color="grey", linestyle="--", linewidth=0.8, label="baseline")
    ax.set_title("Speedup vs k")
    ax.set_xlabel("k (clusters)")
    ax.set_ylabel("Speedup")
    ax.legend()

    plt.tight_layout()
    out = os.path.join(OUT_DIR, "benchmark_plot.png")
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"[INFO] Plot saved → {out}")

    # ── per-iteration time comparison (last k tested) ─────────────────────
    serial_iter = pd.read_csv(os.path.join(OUT_DIR, "serial_iter_times.csv"))
    parallel_iter_file = os.path.join(OUT_DIR, "parallel_iter_times.csv")
    if os.path.exists(parallel_iter_file):
        parallel_iter = pd.read_csv(parallel_iter_file)
        fig2, ax2 = plt.subplots(figsize=(8, 4))
        ax2.plot(serial_iter["iter"],   serial_iter["time_s"],   label="serial",   marker=".")
        ax2.plot(parallel_iter["iter"], parallel_iter["time_s"], label=f"mpi n={MPI_PROCS[-1]}", marker=".")
        ax2.set_title(f"Per-iteration time (k={K_VALUES[-1]})")
        ax2.set_xlabel("Iteration"); ax2.set_ylabel("Seconds")
        ax2.legend()
        plt.tight_layout()
        fig2.savefig(os.path.join(OUT_DIR, "iter_time_comparison.png"), dpi=120)
        plt.close(fig2)

    print("[DONE] Benchmark complete.")


if __name__ == "__main__":
    main()
