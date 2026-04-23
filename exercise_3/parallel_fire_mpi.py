"""
Exercise 3 – Forest Fire Cellular Automaton (Parallel MPI version with mpi4py)

Domain decomposition: the grid is partitioned into horizontal slabs (row blocks).
Each MPI process owns a contiguous band of rows.  At every iteration, ghost rows
are exchanged between neighbouring processes via Sendrecv so that the spread rule
can inspect cross-boundary neighbours without shared memory.

Run example:
    mpiexec -n 4 python parallel_fire_mpi.py --rows 200 --cols 200 --steps 100
"""

import numpy as np
import pandas as pd
import time
import os
import json
import argparse

from mpi4py import MPI

# ─────────────────── automaton parameters (same as serial) ────────────────
BURN_LIFETIME    = 3
SPREAD_PROB_BASE = 0.4
FRP_SCALE        = 0.01
RANDOM_SEED      = 42
VEG_RATIO        = 0.8
FIRMS_API_KEY    = "YOUR_API_KEY_HERE"

REGION = {
    "name": "California_2023",
    "lat_min": 36.0, "lat_max": 40.0,
    "lon_min": -122.0, "lon_max": -118.0,
}


# ─────────────────────── helpers shared with serial ──────────────────────

def _synthetic_hotspots(region, n=60):
    rng = np.random.default_rng(RANDOM_SEED)
    return pd.DataFrame({
        "latitude":  rng.uniform(region["lat_min"], region["lat_max"], n),
        "longitude": rng.uniform(region["lon_min"], region["lon_max"], n),
        "frp":       rng.uniform(10, 200, n),
    })


def build_full_grid(region, rows, cols, hotspots):
    """Construct the initial full grid on rank-0 (then scatter to all ranks)."""
    rng = np.random.default_rng(RANDOM_SEED)
    lat_edges = np.linspace(region["lat_min"], region["lat_max"], rows + 1)
    lon_edges = np.linspace(region["lon_min"], region["lon_max"], cols + 1)

    grid    = rng.choice([0, 1], size=(rows, cols),
                         p=[1 - VEG_RATIO, VEG_RATIO]).astype(np.int8)
    frp_map = np.zeros((rows, cols), dtype=np.float32)

    for _, row in hotspots.iterrows():
        r = int(np.clip(np.searchsorted(lat_edges, row["latitude"]) - 1, 0, rows - 1))
        c = int(np.clip(np.searchsorted(lon_edges, row["longitude"]) - 1, 0, cols - 1))
        if grid[r, c] == 1:
            grid[r, c] = 2
        frp_map[r, c] = max(frp_map[r, c], float(row.get("frp", 50)))

    return grid, frp_map


# ──────────────────────── domain decomposition ────────────────────────────

def decompose(total_rows, size, rank):
    """Return (start_row, local_rows) for this rank."""
    base, rem = divmod(total_rows, size)
    start = rank * base + min(rank, rem)
    local = base + (1 if rank < rem else 0)
    return start, local


# ─────────────────────────── ghost row exchange ───────────────────────────

def exchange_ghosts(local_grid, local_frp, comm, rank, size):
    """
    Each process sends its top and bottom real rows to neighbours and
    receives ghost rows.  Returns (top_ghost, bot_ghost) each of shape (1, cols).
    """
    above = rank - 1 if rank > 0       else MPI.PROC_NULL
    below = rank + 1 if rank < size - 1 else MPI.PROC_NULL

    # ── grid ghosts ──────────────────────────────────────────────────────
    top_send = local_grid[0:1, :].copy()
    bot_send = local_grid[-1:, :].copy()
    top_ghost = np.empty_like(top_send)
    bot_ghost = np.empty_like(bot_send)

    comm.Sendrecv(bot_send, dest=below,  recvbuf=top_ghost, source=above)
    comm.Sendrecv(top_send, dest=above,  recvbuf=bot_ghost, source=below)

    # ── frp ghosts ───────────────────────────────────────────────────────
    top_frp_send = local_frp[0:1, :].copy()
    bot_frp_send = local_frp[-1:, :].copy()
    top_frp_ghost = np.empty_like(top_frp_send)
    bot_frp_ghost = np.empty_like(bot_frp_send)

    comm.Sendrecv(bot_frp_send, dest=below, recvbuf=top_frp_ghost, source=above)
    comm.Sendrecv(top_frp_send, dest=above, recvbuf=bot_frp_ghost, source=below)

    # Replace dummy data from PROC_NULL slots with zeros
    if above == MPI.PROC_NULL:
        top_ghost[:] = 0; top_frp_ghost[:] = 0
    if below == MPI.PROC_NULL:
        bot_ghost[:] = 0; bot_frp_ghost[:] = 0

    return (top_ghost, bot_ghost), (top_frp_ghost, bot_frp_ghost)


# ──────────────────────────── automaton step ──────────────────────────────

def step_local(local_grid, local_frp, burn_age, rng,
               top_ghost, bot_ghost, top_frp_ghost, bot_frp_ghost):
    """Advance the local slab by one step, using ghost rows for boundary neighbours."""

    # Build extended views (ghost + real + ghost)
    ext_grid = np.vstack([top_ghost, local_grid, bot_ghost])
    ext_frp  = np.vstack([top_frp_ghost, local_frp, bot_frp_ghost])
    rows, cols = local_grid.shape

    new_grid = local_grid.copy()
    burning  = (local_grid == 2)

    # transition: burning → burned
    burn_age[burning] += 1
    exhausted = burning & (burn_age >= BURN_LIFETIME)
    new_grid[exhausted] = 3

    # neighbour FRP accumulation (using extended arrays, offset by 1)
    neighbor_frp = np.zeros((rows, cols), dtype=np.float32)
    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        nr_start = 1 + dr          # index into ext arrays (1-based for real rows)
        ext_burn = (ext_grid == 2).astype(np.float32)
        ext_frp_ = ext_frp * ext_burn
        # horizontal shift (within columns)
        if dc == 0:
            neighbor_frp += ext_frp_[nr_start: nr_start + rows, :]
        else:
            col_slice = np.roll(ext_frp_[1: 1 + rows, :], dc, axis=1)
            # mask wrap-around at domain edges (first/last column)
            if dc == 1:
                col_slice[:, 0] = 0
            else:
                col_slice[:, -1] = 0
            neighbor_frp += col_slice

    # vertical neighbours (top/bottom, using extended grid)
    for dr in [-1, 1]:
        ext_burn = (ext_grid == 2).astype(np.float32)
        ext_frp_ = ext_frp * ext_burn
        neighbor_frp += ext_frp_[1 + dr: 1 + dr + rows, :]

    susceptible = (local_grid == 1)
    has_nb      = neighbor_frp > 0
    prob        = np.clip(SPREAD_PROB_BASE + FRP_SCALE * neighbor_frp, 0, 0.95)
    ignite      = susceptible & has_nb & (rng.random((rows, cols)) < prob)
    new_grid[ignite] = 2
    burn_age[ignite] = 0

    return new_grid


# ──────────────────────────── main simulation ─────────────────────────────

def run_parallel(rows, cols, n_steps):
    comm  = MPI.COMM_WORLD
    rank  = comm.Get_rank()
    size  = comm.Get_size()

    # ── rank 0 builds the grid, then scatters ────────────────────────────
    if rank == 0:
        hotspots = _synthetic_hotspots(REGION)
        full_grid, full_frp = build_full_grid(REGION, rows, cols, hotspots)
    else:
        full_grid = full_frp = None

    # Compute row counts and displacements for Scatterv
    counts = []
    displs = []
    for r in range(size):
        s, lr = decompose(rows, size, r)
        counts.append(lr * cols)
        displs.append(s * cols)

    local_rows = decompose(rows, size, rank)[1]
    local_grid = np.empty((local_rows, cols), dtype=np.int8)
    local_frp  = np.empty((local_rows, cols), dtype=np.float32)

    comm.Scatterv([full_grid.flatten() if rank == 0 else None,
                   counts, displs, MPI.SIGNED_CHAR],
                  local_grid.flatten(), root=0)
    comm.Scatterv([full_frp.flatten() if rank == 0 else None,
                   counts, displs, MPI.FLOAT],
                  local_frp.flatten(), root=0)

    burn_age = np.zeros((local_rows, cols), dtype=np.int32)
    rng      = np.random.default_rng(RANDOM_SEED + rank)
    stats    = []

    comm.Barrier()
    t0 = MPI.Wtime()

    for s in range(n_steps):
        ghosts, frp_ghosts = exchange_ghosts(local_grid, local_frp, comm, rank, size)
        local_grid = step_local(local_grid, local_frp, burn_age, rng,
                                ghosts[0], ghosts[1],
                                frp_ghosts[0], frp_ghosts[1])

        # aggregate global statistics
        local_counts = np.array([
            (local_grid == 0).sum(), (local_grid == 1).sum(),
            (local_grid == 2).sum(), (local_grid == 3).sum(),
        ], dtype=np.int64)
        global_counts = np.zeros(4, dtype=np.int64)
        comm.Reduce(local_counts, global_counts, op=MPI.SUM, root=0)
        if rank == 0:
            stats.append({"step": s + 1,
                          "non_burnable": int(global_counts[0]),
                          "susceptible":  int(global_counts[1]),
                          "burning":      int(global_counts[2]),
                          "burned":       int(global_counts[3])})
            if global_counts[2] == 0:
                print(f"[MPI] Fire extinguished at step {s + 1}.")
                break

    comm.Barrier()
    elapsed = MPI.Wtime() - t0

    # ── gather final grid on rank 0 ──────────────────────────────────────
    recv_buf = None
    if rank == 0:
        recv_buf = np.empty(rows * cols, dtype=np.int8)
    comm.Gatherv(local_grid.flatten(),
                 [recv_buf, counts, displs, MPI.SIGNED_CHAR], root=0)

    if rank == 0:
        final_grid = recv_buf.reshape(rows, cols)
        os.makedirs("outputs", exist_ok=True)
        pd.DataFrame(stats).to_csv("outputs/parallel_stats.csv", index=False)
        timing = {
            "mode": "parallel_mpi",
            "processes": size,
            "elapsed_s": elapsed,
            "steps": n_steps,
            "grid": f"{rows}x{cols}",
        }
        with open("outputs/parallel_timing.json", "w") as f:
            json.dump(timing, f, indent=2)
        print(f"[MPI rank 0] Elapsed: {elapsed:.4f} s  (processes={size})")
        print(f"[MPI rank 0] Final burned cells: {int((final_grid == 3).sum())}")


# ────────────────────────────── CLI ──────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows",  type=int, default=200)
    parser.add_argument("--cols",  type=int, default=200)
    parser.add_argument("--steps", type=int, default=100)
    args = parser.parse_args()
    run_parallel(args.rows, args.cols, args.steps)
