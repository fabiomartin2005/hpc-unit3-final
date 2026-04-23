import numpy as np
from multiprocessing import Pool
import time

def worker(args):
    A_chunk, B = args
    return np.dot(A_chunk, B)


def matmul_parallel_rows(A, B, workers=4):
    chunks = np.array_split(A, workers)

    with Pool(workers) as p:
        results = p.map(worker, [(chunk, B) for chunk in chunks])

    return np.vstack(results)


def run(n=500, workers=4):
    np.random.seed(42)
    A = np.random.rand(n, n)
    B = np.random.rand(n, n)

    start = time.time()
    C = matmul_parallel_rows(A, B, workers)
    end = time.time()

    print(f"[ROWS] n={n}, workers={workers}, time={end - start:.4f}s")
    return end - start


if __name__ == "__main__":
    run()