import numpy as np
from multiprocessing import Pool
import time

def block_multiply(args):
    A_block, B = args
    return np.dot(A_block, B)


def matmul_parallel_blocks(A, B, workers=4):
    blocks = np.array_split(A, workers)

    with Pool(workers) as p:
        results = p.map(block_multiply, [(block, B) for block in blocks])

    return np.vstack(results)


def run(n=500, workers=4):
    np.random.seed(42)
    A = np.random.rand(n, n)
    B = np.random.rand(n, n)

    start = time.time()
    C = matmul_parallel_blocks(A, B, workers)
    end = time.time()

    print(f"[BLOCKS] n={n}, workers={workers}, time={end - start:.4f}s")
    return end - start


if __name__ == "__main__":
    run()