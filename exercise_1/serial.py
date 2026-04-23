import numpy as np
import time

def matmul_serial(A, B):
    n, m = A.shape
    m2, p = B.shape
    assert m == m2

    C = np.zeros((n, p))
    for i in range(n):
        for j in range(p):
            for k in range(m):
                C[i, j] += A[i, k] * B[k, j]
    return C


def run(n=300):
    np.random.seed(42)
    A = np.random.rand(n, n)
    B = np.random.rand(n, n)

    start = time.time()
    C = matmul_serial(A, B)
    end = time.time()

    print(f"[SERIAL] n={n} time={end - start:.4f}s")
    return end - start


if __name__ == "__main__":
    run()