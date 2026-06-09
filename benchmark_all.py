"""WebGPU structure factor benchmark."""

import numpy as np
import time
from glsf import sf3d


def make_data(N, K, L=10.0):
    k_factor = 2 * np.pi / L
    rng = np.random.default_rng(42)
    points = rng.uniform(-L / 2, L / 2, (N, 3)).astype(np.float32)

    nx = np.arange(-K, K + 1)
    ny = np.arange(-K, K + 1)
    nz = np.arange(0, K + 1)
    NX, NY, NZ = np.meshgrid(nx, ny, nz, indexing="ij")
    k_vecs = (
        np.stack([NX.flatten(), NY.flatten(), NZ.flatten()], axis=1).astype(np.float32)
        * k_factor
    )
    return points, k_vecs


def benchmark(points, k_vecs):
    sf3d(points, k_vecs)  # warmup
    start = time.perf_counter()
    sf3d(points, k_vecs)
    return (time.perf_counter() - start) * 1000


if __name__ == "__main__":
    for N, K in [(1000, 10), (10000, 10), (10000, 32), (1000, 64), (10000, 64)]:
        points, k_vecs = make_data(N, K)
        Nk = len(k_vecs)
        t = benchmark(points, k_vecs)
        print(f"N={N:>6}, Nk={Nk:>8}: {t:>10.2f} ms")
