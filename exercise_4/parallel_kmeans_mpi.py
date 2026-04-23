"""
Exercise 4 – Parallel K-Means with mpi4py

Parallelisation strategy
─────────────────────────
1. Rank 0 loads and broadcasts initial centroids and dataset shape.
2. The dataset is partitioned row-wise: each rank loads / receives its local
   shard of X.
3. Every iteration:
     a. Each rank computes assignments and accumulates local sums + counts per cluster.
     b. MPI Allreduce (SUM) aggregates partial sums across all ranks → global centroids.
     c. Convergence is checked collectively.
4. Rank 0 writes timing and result files.

Run example:
    mpiexec -n 4 python parallel_kmeans_mpi.py --k 7 --max_rows 100000
"""

import numpy as np
import pandas as pd
import time
import os
import json
import argparse

from mpi4py import MPI

# ─── reuse data loading and init from the serial module ──────────────────
import sys
sys.path.insert(0, os.path.dirname(__file__))
from serial_kmeans import (load_covertype, kmeans_init,
                            N_CLUSTERS, MAX_ITER, TOL, RANDOM_SEED)


# ────────────────────── parallel helpers ─────────────────────────────────

def scatter_data(X: np.ndarray, comm, rank, size) -> np.ndarray:
    """Scatter rows of X evenly across processes."""
    n, d = X.shape
    counts   = [n // size + (1 if i < n % size else 0) for i in range(size)]
    displs   = [sum(counts[:i]) for i in range(size)]
    local_n  = counts[rank]
    local_X  = np.empty((local_n, d), dtype=np.float64)

    # Use Scatterv with flattened arrays
    send_buf = X.flatten() if rank == 0 else None
    flat_counts  = [c * d for c in counts]
    flat_displs  = [s * d for s in displs]
    comm.Scatterv([send_buf, flat_counts, flat_displs, MPI.DOUBLE],
                  local_X.flatten(), root=0)
    return local_X


def local_assign_accumulate(local_X, centers):
    """
    Compute local assignments and partial cluster sums/counts.

    Returns
    -------
    labels    : (local_n,) int array
    local_sum : (k, d) sum of points per cluster
    local_cnt : (k,) count of points per cluster
    """
    k = len(centers)
    d = local_X.shape[1]
    diffs  = local_X[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists2 = np.sum(diffs ** 2, axis=2)
    labels = np.argmin(dists2, axis=1)

    local_sum = np.zeros((k, d), dtype=np.float64)
    local_cnt = np.zeros(k, dtype=np.float64)
    for j in range(k):
        mask = labels == j
        if mask.any():
            local_sum[j] = local_X[mask].sum(axis=0)
            local_cnt[j] = mask.sum()
    return labels, local_sum, local_cnt


def aggregate_centers(local_sum, local_cnt, old_centers, comm):
    """Allreduce partial sums and counts → new global centroids."""
    k, d = local_sum.shape
    global_sum = np.zeros_like(local_sum)
    global_cnt = np.zeros_like(local_cnt)
    comm.Allreduce(local_sum, global_sum, op=MPI.SUM)
    comm.Allreduce(local_cnt, global_cnt, op=MPI.SUM)

    new_centers = old_centers.copy()
    for j in range(k):
        if global_cnt[j] > 0:
            new_centers[j] = global_sum[j] / global_cnt[j]
    return new_centers


# ──────────────────────── main parallel run ───────────────────────────────

def kmeans_parallel(X_full: np.ndarray, k: int = N_CLUSTERS,
                    max_iter: int = MAX_ITER, tol: float = TOL,
                    seed: int = RANDOM_SEED) -> dict:
    comm  = MPI.COMM_WORLD
    rank  = comm.Get_rank()
    size  = comm.Get_size()

    # Broadcast data shape
    shape = np.array(X_full.shape if rank == 0 else [0, 0], dtype=np.int64)
    comm.Bcast(shape, root=0)
    if rank != 0:
        X_full = np.empty(shape, dtype=np.float64)
    comm.Bcast(X_full, root=0)

    # Scatter local shard
    local_X = scatter_data(X_full, comm, rank, size)

    # Initialise centers on rank 0 and broadcast
    if rank == 0:
        centers = kmeans_init(X_full, k, seed)
    else:
        centers = np.empty((k, X_full.shape[1]), dtype=np.float64)
    comm.Bcast(centers, root=0)

    iter_times = []
    n_iter     = 0
    labels_all = np.zeros(X_full.shape[0], dtype=np.int32)

    comm.Barrier()
    t_total = MPI.Wtime()

    for i in range(max_iter):
        t0 = MPI.Wtime()

        local_labels, local_sum, local_cnt = local_assign_accumulate(local_X, centers)
        new_centers = aggregate_centers(local_sum, local_cnt, centers, comm)
        shift = float(np.max(np.linalg.norm(new_centers - centers, axis=1)))

        # broadcast convergence flag
        converged = int(shift < tol)
        converged = comm.allreduce(converged, op=MPI.MIN)

        centers = new_centers
        iter_times.append(MPI.Wtime() - t0)
        n_iter += 1

        if rank == 0:
            print(f"  iter {i+1:3d}  shift={shift:.2e}  time={iter_times[-1]:.4f}s")

        if converged:
            if rank == 0:
                print(f"[INFO] Converged at iteration {i + 1}.")
            break

    comm.Barrier()
    total_s = MPI.Wtime() - t_total

    # Gather all labels on rank 0
    n, d    = X_full.shape
    n_per   = [n // size + (1 if i < n % size else 0) for i in range(size)]
    disp    = [sum(n_per[:i]) for i in range(size)]
    recv    = np.empty(n, dtype=np.int32) if rank == 0 else None
    comm.Gatherv(local_labels, [recv, n_per, disp, MPI.INT], root=0)

    # Inertia on rank 0
    inertia = 0.0
    if rank == 0:
        labels_all = recv
        for j in range(k):
            mask = labels_all == j
            if mask.any():
                inertia += float(np.sum((X_full[mask] - centers[j]) ** 2))

    return {
        "centers":    centers,
        "labels":     labels_all if rank == 0 else None,
        "inertia":    inertia,
        "iter_times": iter_times,
        "n_iter":     n_iter,
        "total_s":    total_s,
        "rank":       rank,
        "size":       size,
    }


# ────────────────────────────── CLI ──────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--k",        type=int, default=N_CLUSTERS)
    parser.add_argument("--max_iter", type=int, default=MAX_ITER)
    parser.add_argument("--max_rows", type=int, default=100_000)
    args = parser.parse_args()

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    if rank == 0:
        print("=== Exercise 4 – Parallel K-Means (MPI) ===")
        X, y, *_ = load_covertype(max_rows=args.max_rows)
    else:
        X = None

    result = kmeans_parallel(X, k=args.k, max_iter=args.max_iter)

    if rank == 0:
        print(f"\n[RESULT] Total time:  {result['total_s']:.4f} s")
        print(f"[RESULT] Iterations:  {result['n_iter']}")
        print(f"[RESULT] Inertia:     {result['inertia']:.2f}")
        print(f"[RESULT] Processes:   {result['size']}")

        os.makedirs("outputs", exist_ok=True)
        timing = {
            "mode":        "parallel_mpi",
            "processes":   result["size"],
            "k":           args.k,
            "n_iter":      result["n_iter"],
            "total_s":     result["total_s"],
            "mean_iter_s": float(np.mean(result["iter_times"])),
            "inertia":     result["inertia"],
        }
        with open("outputs/parallel_timing.json", "w") as f:
            json.dump(timing, f, indent=2)

        pd.DataFrame({"iter": range(1, len(result["iter_times"]) + 1),
                      "time_s": result["iter_times"]}
                     ).to_csv("outputs/parallel_iter_times.csv", index=False)

        np.save("outputs/parallel_labels.npy",  result["labels"])
        np.save("outputs/parallel_centers.npy", result["centers"])
        print("[DONE] Parallel run complete.")
