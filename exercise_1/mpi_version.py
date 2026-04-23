from mpi4py import MPI
import numpy as np
import time

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

n = 500

if rank == 0:
    np.random.seed(42)
    A = np.random.rand(n, n)
    B = np.random.rand(n, n)
else:
    A = None
    B = np.random.rand(n, n)

# Scatter A
A_chunks = None
if rank == 0:
    A_chunks = np.array_split(A, size)

local_A = comm.scatter(A_chunks, root=0)

# Broadcast B
B = comm.bcast(B, root=0)

comm.Barrier()
start = time.time()

local_C = np.dot(local_A, B)

C = comm.gather(local_C, root=0)

comm.Barrier()
end = time.time()

if rank == 0:
    C = np.vstack(C)
    print(f"[MPI] processes={size}, time={end - start:.4f}s")