import numpy as np
import torch
import mlx.core as mx
import glsf
import time

CHUNK_SIZE = 4096

def benchmark_numpy(points, k_vecs):
    N = len(points)
    start = time.perf_counter()
    sk = np.empty(len(k_vecs))
    for i in range(0, len(k_vecs), CHUNK_SIZE):
        k_chunk = k_vecs[i:i + CHUNK_SIZE]
        phases = k_chunk @ points.T
        c = np.cos(phases).sum(axis=1)
        s = np.sin(phases).sum(axis=1)
        sk[i:i + len(k_chunk)] = (c**2 + s**2) / N
    return (time.perf_counter() - start) * 1000

def benchmark_torch_gpu(points, k_vecs):
    N = len(points)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    p_t = torch.from_numpy(points).to(device)
    k_t = torch.from_numpy(k_vecs).to(device)

    # Warmup
    _ = torch.cos(torch.matmul(k_t[:10], p_t.T)).sum(dim=1)
    torch.mps.synchronize()

    start = time.perf_counter()
    sk_parts = []
    for i in range(0, len(k_vecs), CHUNK_SIZE):
        k_chunk = k_t[i:i + CHUNK_SIZE]
        phases = torch.matmul(k_chunk, p_t.T)
        c = torch.cos(phases).sum(dim=1)
        s = torch.sin(phases).sum(dim=1)
        sk_parts.append((c**2 + s**2) / N)
    sk = torch.cat(sk_parts)
    torch.mps.synchronize()
    return (time.perf_counter() - start) * 1000

@mx.compile
def mlx_chunk(p, k_chunk, N):
    phases = mx.matmul(k_chunk, p.T)
    c = mx.sum(mx.cos(phases), axis=1)
    s = mx.sum(mx.sin(phases), axis=1)
    return (c**2 + s**2) / N

def benchmark_mlx_gpu(points, k_vecs):
    N = len(points)
    p_m = mx.array(points)
    k_m = mx.array(k_vecs)

    # Warmup
    _ = mlx_chunk(p_m, k_m[:10], N)
    mx.eval(_)

    start = time.perf_counter()
    sk_parts = []
    for i in range(0, len(k_vecs), CHUNK_SIZE):
        k_chunk = k_m[i:i + CHUNK_SIZE]
        sk_parts.append(mlx_chunk(p_m, k_chunk, N))
    sk = mx.concatenate(sk_parts)
    mx.eval(sk)
    return (time.perf_counter() - start) * 1000

def benchmark_sf3d_gpu(points, k_vecs):
    # Our custom OpenCL package
    # Warmup (handled by static RAII engine in C++)
    _ = glsf.sf3d(points, k_vecs)

    start = time.perf_counter()
    sk = glsf.sf3d(points, k_vecs)
    return (time.perf_counter() - start) * 1000

def run_suite(N, K):
    L = 10.0
    k_factor = 2 * np.pi / L
    points = np.random.uniform(-L/2, L/2, (N, 3)).astype(np.float32)
    
    nx = np.arange(-K, K + 1)
    ny = np.arange(-K, K + 1)
    nz = np.arange(0, K + 1)
    NX, NY, NZ = np.meshgrid(nx, ny, nz, indexing='ij')
    k_vecs = np.stack([NX.flatten(), NY.flatten(), NZ.flatten()], axis=1).astype(np.float32) * k_factor
    Nk = len(k_vecs)

    print(f"\n--- Benchmark: N={N}, Nk={Nk} ---")
    
    # 1. NumPy (Chunked, but still slow for very large systems)
    if N * Nk < 2e8:
        t_np = benchmark_numpy(points, k_vecs)
        print(f"NumPy:         {t_np:>10.2f} ms")
    else:
        print(f"NumPy:              SKIPPED (Too slow)")

    # 2. PyTorch GPU
    try:
        t_torch = benchmark_torch_gpu(points, k_vecs)
        print(f"PyTorch (GPU): {t_torch:>10.2f} ms")
    except Exception as e:
        print(f"PyTorch (GPU):      FAILED ({e})")

    # 3. MLX GPU
    try:
        t_mlx = benchmark_mlx_gpu(points, k_vecs)
        print(f"MLX (GPU):     {t_mlx:>10.2f} ms")
    except Exception as e:
        print(f"MLX (GPU):          FAILED ({e})")

    # 4. Our OpenCL Package
    t_sf3d = benchmark_sf3d_gpu(points, k_vecs)
    print(f"sf3d-gpu:      {t_sf3d:>10.2f} ms")

if __name__ == "__main__":
    print("Starting Unified Benchmark Suite...")
    # Small System
    run_suite(1000, 10)
    # Production System
    run_suite(10000, 10)
    # Large System (Will OOM NumPy/MLX/Torch if not careful)
    run_suite(10000, 32)
