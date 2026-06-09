import numpy as np
import glsf
import time

def test():
    N = 10000
    L = 10.0
    K = 10
    
    np.random.seed(42)
    points = np.random.uniform(-L/2, L/2, (N, 3)).astype(np.float32)
    
    k_factor = 2 * np.pi / L
    nx = np.arange(-K, K + 1)
    ny = np.arange(-K, K + 1)
    nz = np.arange(0, K + 1)
    NX, NY, NZ = np.meshgrid(nx, ny, nz, indexing='ij')
    k_vecs = np.stack([NX.flatten(), NY.flatten(), NZ.flatten()], axis=1).astype(np.float32) * k_factor
    
    print(f"Testing glsf package (N={N}, Nk={len(k_vecs)})...")
    
    # First call (GPU init)
    start = time.perf_counter()
    sk = glsf.sf3d(points, k_vecs)
    end = time.perf_counter()
    print(f"First call: {(end - start)*1000:.3f} ms")

    # Second call (Warm context)
    start = time.perf_counter()
    sk = glsf.sf3d(points, k_vecs)
    end = time.perf_counter()
    print(f"Second call: {(end - start)*1000:.3f} ms")
    
    print(f"Result mean: {np.mean(sk):.5f} (Expect ~1.0 now that k=0 is masked)")
    
    # Check consistency
    if np.abs(np.mean(sk) - 1.0) < 0.1:
        print("SUCCESS: Package is working correctly and k=0 is masked!")
    else:
        print("FAILURE: Result mean is unexpected.")

if __name__ == "__main__":
    test()
