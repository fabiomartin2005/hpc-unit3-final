"""
Exercise 3 - Forest Fire Cellular Automaton (Serial Version)
Uses NASA FIRMS hotspot data to initialize a 2D grid and simulate fire propagation.

States:
  0 = non-burnable / outside domain
  1 = susceptible vegetation
  2 = burning
  3 = burned
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.animation as animation
import requests
import time
import os
import json
from datetime import datetime

# ─────────────────────────── configuration ────────────────────────────────
REGION = {
    "name": "California_2023",
    "lat_min": 36.0, "lat_max": 40.0,
    "lon_min": -122.0, "lon_max": -118.0,
}
GRID_ROWS = 200
GRID_COLS = 200
N_STEPS = 100
BURN_LIFETIME = 3          # steps a cell stays burning before becoming burned
SPREAD_PROB_BASE = 0.4     # base probability of spreading to a susceptible neighbor
FRP_SCALE = 0.01           # how much fire radiative power boosts spread probability
RANDOM_SEED = 42
FIRMS_API_KEY = "YOUR_API_KEY_HERE"   # replace or set env var FIRMS_API_KEY


# ────────────────────────── data acquisition ──────────────────────────────

def fetch_firms_data(region: dict, api_key: str, days: int = 7) -> pd.DataFrame:
    """
    Download MODIS/VIIRS hotspot detections from NASA FIRMS for the given region.
    Falls back to synthetic data if the key is a placeholder or the request fails.
    """
    key = os.environ.get("FIRMS_API_KEY", api_key)
    url = (
        f"https://firms.modaps.eosdis.nasa.gov/api/area/csv/{key}/MODIS_NRT/"
        f"{region['lon_min']},{region['lat_min']},{region['lon_max']},{region['lat_max']}"
        f"/{days}"
    )
    if key == "YOUR_API_KEY_HERE":
        print("[INFO] No FIRMS API key detected – generating synthetic hotspot data.")
        return _synthetic_hotspots(region)

    try:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        from io import StringIO
        df = pd.read_csv(StringIO(resp.text))
        print(f"[INFO] Downloaded {len(df)} FIRMS detections.")
        return df
    except Exception as exc:
        print(f"[WARN] FIRMS request failed ({exc}). Using synthetic data.")
        return _synthetic_hotspots(region)


def _synthetic_hotspots(region: dict, n: int = 60) -> pd.DataFrame:
    """Generate synthetic hotspot detections uniformly spread over the region."""
    rng = np.random.default_rng(RANDOM_SEED)
    return pd.DataFrame({
        "latitude":  rng.uniform(region["lat_min"], region["lat_max"], n),
        "longitude": rng.uniform(region["lon_min"], region["lon_max"], n),
        "frp":       rng.uniform(10, 200, n),      # fire radiative power (MW)
    })


# ────────────────────────── grid construction ─────────────────────────────

def build_grid(region: dict, rows: int, cols: int,
               hotspots: pd.DataFrame, veg_ratio: float = 0.8) -> tuple:
    """
    Create the initial automaton grid and a FRP map.

    Returns
    -------
    grid : np.ndarray (rows × cols) with states 0/1/2
    frp_map : np.ndarray (rows × cols) fire radiative power at each cell
    lat_edges, lon_edges : coordinate vectors
    """
    rng = np.random.default_rng(RANDOM_SEED)

    lat_edges = np.linspace(region["lat_min"], region["lat_max"], rows + 1)
    lon_edges = np.linspace(region["lon_min"], region["lon_max"], cols + 1)

    # start with mostly vegetated cells
    grid = rng.choice([0, 1], size=(rows, cols),
                      p=[1 - veg_ratio, veg_ratio]).astype(np.int8)

    frp_map = np.zeros((rows, cols), dtype=np.float32)

    # map hotspots onto grid
    for _, row in hotspots.iterrows():
        r = np.searchsorted(lat_edges, row["latitude"]) - 1
        c = np.searchsorted(lon_edges, row["longitude"]) - 1
        r = np.clip(r, 0, rows - 1)
        c = np.clip(c, 0, cols - 1)
        if grid[r, c] == 1:
            grid[r, c] = 2        # ignite
        frp_map[r, c] = max(frp_map[r, c], row.get("frp", 50))

    return grid, frp_map, lat_edges, lon_edges


# ────────────────────────── automaton step ────────────────────────────────

def step(grid: np.ndarray, frp_map: np.ndarray,
         burn_age: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """
    Advance the cellular automaton by one time step.

    Parameters
    ----------
    grid      : current state grid (modified in place)
    frp_map   : fire radiative power per cell
    burn_age  : how many steps each cell has been burning
    rng       : random number generator
    """
    new_grid = grid.copy()

    burning_mask = (grid == 2)

    # ── transition burning → burned ────────────────────────────────────────
    burn_age[burning_mask] += 1
    exhausted = burning_mask & (burn_age >= BURN_LIFETIME)
    new_grid[exhausted] = 3

    # ── spread fire to susceptible neighbors ───────────────────────────────
    # build count of burning neighbors using convolution-like shift
    neighbor_frp = np.zeros_like(frp_map)
    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        shifted = np.roll(np.roll(burning_mask.astype(np.float32), dr, axis=0), dc, axis=1)
        frp_shifted = np.roll(np.roll(frp_map * burning_mask, dr, axis=0), dc, axis=1)
        neighbor_frp += frp_shifted

    susceptible = (grid == 1)
    has_burning_neighbor = neighbor_frp > 0
    spread_prob = np.clip(SPREAD_PROB_BASE + FRP_SCALE * neighbor_frp, 0, 0.95)
    ignite = susceptible & has_burning_neighbor & (rng.random(grid.shape) < spread_prob)
    new_grid[ignite] = 2
    burn_age[ignite] = 0

    return new_grid


# ────────────────────────── simulation driver ─────────────────────────────

def run_serial(grid_init: np.ndarray, frp_map: np.ndarray,
               n_steps: int, save_snapshots: bool = True) -> tuple:
    """
    Run the serial simulation.

    Returns
    -------
    final_grid : state after n_steps
    snapshots  : list of (step, grid copy) for selected steps
    stats      : list of dicts with per-step counts
    elapsed    : total wall-clock seconds
    """
    rng = np.random.default_rng(RANDOM_SEED)
    grid = grid_init.copy()
    burn_age = np.zeros_like(grid, dtype=np.int32)
    snapshots = []
    stats = []
    snap_steps = set(np.linspace(0, n_steps - 1, min(10, n_steps), dtype=int))

    t0 = time.perf_counter()
    for s in range(n_steps):
        grid = step(grid, frp_map, burn_age, rng)
        counts = {
            "step": s + 1,
            "non_burnable": int(np.sum(grid == 0)),
            "susceptible":  int(np.sum(grid == 1)),
            "burning":      int(np.sum(grid == 2)),
            "burned":       int(np.sum(grid == 3)),
        }
        stats.append(counts)
        if save_snapshots and s in snap_steps:
            snapshots.append((s + 1, grid.copy()))
        if np.sum(grid == 2) == 0:
            print(f"[INFO] Fire extinguished at step {s + 1}.")
            break
    elapsed = time.perf_counter() - t0
    return grid, snapshots, stats, elapsed


# ────────────────────────── visualisation ────────────────────────────────

CMAP = mcolors.ListedColormap(["#1a1a2e", "#2d6a4f", "#e76f51", "#6b4226"])
NORM = mcolors.BoundaryNorm([0, 1, 2, 3, 4], CMAP.N)


def plot_snapshots(snapshots: list, out_dir: str = "outputs") -> None:
    os.makedirs(out_dir, exist_ok=True)
    n = len(snapshots)
    cols = min(5, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
    axes = np.array(axes).flatten()
    for ax, (step_n, grid) in zip(axes, snapshots):
        ax.imshow(grid, cmap=CMAP, norm=NORM, origin="lower")
        ax.set_title(f"Step {step_n}", fontsize=9)
        ax.axis("off")
    for ax in axes[n:]:
        ax.axis("off")
    labels = ["Non-burnable", "Susceptible", "Burning", "Burned"]
    handles = [plt.Rectangle((0, 0), 1, 1, color=CMAP(i / 3)) for i in range(4)]
    fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=8)
    plt.tight_layout()
    path = os.path.join(out_dir, "serial_snapshots.png")
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"[INFO] Snapshots saved → {path}")


def plot_stats(stats: list, out_dir: str = "outputs") -> None:
    os.makedirs(out_dir, exist_ok=True)
    df = pd.DataFrame(stats)
    fig, ax = plt.subplots(figsize=(8, 4))
    for col, color in [("susceptible", "#2d6a4f"), ("burning", "#e76f51"),
                        ("burned", "#6b4226")]:
        ax.plot(df["step"], df[col], label=col.capitalize(), color=color)
    ax.set_xlabel("Step")
    ax.set_ylabel("Number of cells")
    ax.set_title("Fire dynamics over time (serial)")
    ax.legend()
    plt.tight_layout()
    path = os.path.join(out_dir, "serial_stats.png")
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"[INFO] Stats plot saved → {path}")


# ────────────────────────── entry point ──────────────────────────────────

def main():
    os.makedirs("outputs", exist_ok=True)

    print("=== Exercise 3 – Serial Forest Fire Automaton ===")
    hotspots = fetch_firms_data(REGION, FIRMS_API_KEY)
    hotspots.to_csv("outputs/hotspots.csv", index=False)

    print(f"[INFO] Building {GRID_ROWS}×{GRID_COLS} grid…")
    grid, frp_map, lat_edges, lon_edges = build_grid(
        REGION, GRID_ROWS, GRID_COLS, hotspots)

    n_ignited = int(np.sum(grid == 2))
    print(f"[INFO] Ignition points from FIRMS: {n_ignited}")

    print(f"[INFO] Running serial simulation ({N_STEPS} steps)…")
    final_grid, snapshots, stats, elapsed = run_serial(grid, frp_map, N_STEPS)

    print(f"[RESULT] Serial elapsed: {elapsed:.4f} s")
    print(f"[RESULT] Final burned cells: {int(np.sum(final_grid == 3))}")

    # persist results
    pd.DataFrame(stats).to_csv("outputs/serial_stats.csv", index=False)
    timing = {"mode": "serial", "elapsed_s": elapsed,
              "steps": N_STEPS, "grid": f"{GRID_ROWS}x{GRID_COLS}"}
    with open("outputs/serial_timing.json", "w") as f:
        json.dump(timing, f, indent=2)

    plot_snapshots(snapshots)
    plot_stats(stats)
    print("[DONE] Serial run complete.")


if __name__ == "__main__":
    main()
