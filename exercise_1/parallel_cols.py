import numpy as np
from multiprocessing import Pool
import time

def worker(args):
    A, B_chunk = args
    return np.dot(A, B_chunk)


def matmul_parallel_cols(A, B, workers=4):
    chunks = np.array_split(B, workers, axis=1)

    with Pool(workers) as p:
        results = p.map(worker, [(A, chunk) for chunk in chunks])

    return np.hstack(results)


def run(n=500, workers=4):
    np.random.seed(42)
    A = np.random.rand(n, n)
    B = np.random.rand(n, n)

    start = time.time()
    C = matmul_parallel_cols(A, B, workers)
    end = time.time()

    print(f"[COLS] n={n}, workers={workers}, time={end - start:.4f}s")
    return end - start


if __name__ == "__main__":
    run()