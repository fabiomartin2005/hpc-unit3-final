"""
Exercise 4 – Parallel K-Means Clustering (Serial Baseline)

Dataset: Covertype (UCI ML Repository)
  https://archive.ics.uci.edu/ml/datasets/covertype

The script:
  1. Downloads / loads the Covertype dataset.
  2. Preprocesses features (standardise numeric columns).
  3. Runs a serial K-Means and measures per-iteration and total runtime.
  4. Saves timing, cluster statistics, and a 2-D PCA visualisation.
"""

import numpy as np
import pandas as pd
import time
import os
import json
import argparse
import urllib.request

# ───────────────────────── configuration ─────────────────────────────────
N_CLUSTERS  = 7        # matches the 7 true forest-cover types
MAX_ITER    = 50
TOL         = 1e-4     # centroid shift convergence threshold
RANDOM_SEED = 42
DATA_PATH   = "data/covtype.csv"
DATA_URL    = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/"
    "covtype/covtype.data.gz"
)

FEATURE_NAMES = [
    "Elevation", "Aspect", "Slope",
    "H_Dist_Hydro", "V_Dist_Hydro", "H_Dist_Road",
    "Hillshade_9am", "Hillshade_Noon", "Hillshade_3pm",
    "H_Dist_Fire",
    *[f"Wilderness_{i}" for i in range(4)],
    *[f"SoilType_{i}"   for i in range(40)],
    "CoverType",
]


# ────────────────────────── data loading ─────────────────────────────────

def load_covertype(path: str = DATA_PATH, max_rows: int = None) -> np.ndarray:
    """
    Load the Covertype dataset. Downloads it if not found locally.
    Returns the feature matrix X (float64, standardised) and labels y.
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    if not os.path.exists(path):
        print(f"[INFO] Downloading Covertype dataset → {path}…")
        import gzip, shutil
        gz_path = path + ".gz"
        urllib.request.urlretrieve(DATA_URL, gz_path)
        with gzip.open(gz_path, "rb") as f_in, open(path, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)
        os.remove(gz_path)
        print("[INFO] Download complete.")

    df = pd.read_csv(path, header=None, names=FEATURE_NAMES,
                     nrows=max_rows)
    y = df["CoverType"].values
    X = df.drop(columns=["CoverType"]).values.astype(np.float64)

    # standardise only the first 10 continuous features; binary features stay 0/1
    means = X[:, :10].mean(axis=0)
    stds  = X[:, :10].std(axis=0)
    stds[stds == 0] = 1
    X[:, :10] = (X[:, :10] - means) / stds

    print(f"[INFO] Covertype loaded: {X.shape[0]} samples, {X.shape[1]} features.")
    return X, y, means, stds


# ─────────────────────── K-Means implementation ──────────────────────────

def kmeans_init(X: np.ndarray, k: int, seed: int) -> np.ndarray:
    """K-Means++ initialisation."""
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    idx = rng.integers(n)
    centers = [X[idx]]
    for _ in range(k - 1):
        dists = np.min(
            np.stack([np.sum((X - c) ** 2, axis=1) for c in centers]), axis=0
        )
        probs = dists / dists.sum()
        idx   = rng.choice(n, p=probs)
        centers.append(X[idx])
    return np.array(centers)


def assign(X: np.ndarray, centers: np.ndarray) -> np.ndarray:
    """Assign each sample to the nearest centroid (vectorised)."""
    diffs  = X[:, np.newaxis, :] - centers[np.newaxis, :, :]   # (N, k, d)
    dists2 = np.sum(diffs ** 2, axis=2)                         # (N, k)
    return np.argmin(dists2, axis=1)


def update_centers(X: np.ndarray, labels: np.ndarray, k: int,
                   old_centers: np.ndarray) -> np.ndarray:
    """Recompute centroids; keep old centroid for empty clusters."""
    centers = old_centers.copy()
    for j in range(k):
        mask = labels == j
        if mask.any():
            centers[j] = X[mask].mean(axis=0)
    return centers


def kmeans_serial(X: np.ndarray, k: int = N_CLUSTERS,
                  max_iter: int = MAX_ITER, tol: float = TOL,
                  seed: int = RANDOM_SEED) -> dict:
    """
    Run serial K-Means.

    Returns a dict with:
      centers   – final centroids
      labels    – cluster assignment per sample
      inertia   – within-cluster sum of squares
      iter_times – per-iteration wall-clock times (s)
      n_iter    – actual iterations run
      total_s   – total elapsed time
    """
    centers    = kmeans_init(X, k, seed)
    labels     = np.zeros(X.shape[0], dtype=np.int32)
    iter_times = []
    n_iter     = 0

    t_total = time.perf_counter()

    for i in range(max_iter):
        t0 = time.perf_counter()
        labels      = assign(X, centers)
        new_centers = update_centers(X, labels, k, centers)
        shift       = np.max(np.linalg.norm(new_centers - centers, axis=1))
        centers     = new_centers
        iter_times.append(time.perf_counter() - t0)
        n_iter += 1
        if shift < tol:
            print(f"[INFO] Converged at iteration {i + 1} (shift={shift:.2e}).")
            break

    total_s = time.perf_counter() - t_total

    inertia = sum(
        np.sum((X[labels == j] - centers[j]) ** 2)
        for j in range(k)
        if (labels == j).any()
    )

    return {
        "centers":    centers,
        "labels":     labels,
        "inertia":    float(inertia),
        "iter_times": iter_times,
        "n_iter":     n_iter,
        "total_s":    total_s,
    }


# ─────────────────────────── reporting ──────────────────────────────────

def report(result: dict, k: int, X: np.ndarray) -> None:
    print(f"\n{'='*50}")
    print(f"Serial K-Means  k={k}")
    print(f"  Iterations:  {result['n_iter']}")
    print(f"  Total time:  {result['total_s']:.4f} s")
    print(f"  Mean iter:   {np.mean(result['iter_times']):.4f} s")
    print(f"  Inertia:     {result['inertia']:.2f}")
    print(f"{'='*50}\n")

    labels = result["labels"]
    sizes  = [(labels == j).sum() for j in range(k)]
    for j, sz in enumerate(sizes):
        print(f"  Cluster {j}: {sz} samples ({100*sz/len(labels):.1f} %)")


def save_results(result: dict, k: int, out_dir: str = "outputs") -> None:
    os.makedirs(out_dir, exist_ok=True)
    timing = {
        "mode":       "serial",
        "k":          k,
        "n_iter":     result["n_iter"],
        "total_s":    result["total_s"],
        "mean_iter_s":float(np.mean(result["iter_times"])),
        "inertia":    result["inertia"],
    }
    with open(os.path.join(out_dir, "serial_timing.json"), "w") as f:
        json.dump(timing, f, indent=2)

    pd.DataFrame({"iter": range(1, len(result["iter_times"]) + 1),
                  "time_s": result["iter_times"]}
                 ).to_csv(os.path.join(out_dir, "serial_iter_times.csv"), index=False)

    np.save(os.path.join(out_dir, "serial_labels.npy"), result["labels"])
    np.save(os.path.join(out_dir, "serial_centers.npy"), result["centers"])


def plot_pca(X: np.ndarray, labels: np.ndarray, centers: np.ndarray,
             out_dir: str = "outputs") -> None:
    try:
        import matplotlib.pyplot as plt
        from sklearn.decomposition import PCA
    except ImportError:
        print("[WARN] matplotlib / sklearn not available – skipping PCA plot.")
        return

    pca  = PCA(n_components=2, random_state=RANDOM_SEED)
    X2   = pca.fit_transform(X)
    c2   = pca.transform(centers)

    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(X2[:, 0], X2[:, 1], c=labels, cmap="tab10",
                         s=1, alpha=0.4)
    ax.scatter(c2[:, 0], c2[:, 1], c="black", marker="X", s=120,
               zorder=5, label="Centroids")
    ax.set_title("K-Means clusters (PCA projection, serial)")
    ax.set_xlabel("PC 1"); ax.set_ylabel("PC 2")
    ax.legend()
    plt.colorbar(scatter, ax=ax, label="Cluster ID")
    plt.tight_layout()
    path = os.path.join(out_dir, "serial_pca.png")
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"[INFO] PCA plot saved → {path}")


# ──────────────────────────── CLI ────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--k",        type=int, default=N_CLUSTERS)
    parser.add_argument("--max_iter", type=int, default=MAX_ITER)
    parser.add_argument("--max_rows", type=int, default=100_000,
                        help="Sub-sample for quick testing (None = full dataset)")
    parser.add_argument("--no_plot",  action="store_true")
    args = parser.parse_args()

    print("=== Exercise 4 – Serial K-Means ===")
    X, y, *_ = load_covertype(max_rows=args.max_rows)

    result = kmeans_serial(X, k=args.k, max_iter=args.max_iter)
    report(result, args.k, X)
    save_results(result, args.k)

    if not args.no_plot:
        plot_pca(X, result["labels"], result["centers"])

    print("[DONE] Serial run complete.")
