"""Tests for glsf against a float64 numpy reference.

Validates numerical accuracy, crystal selection rules, edge cases,
and statistical properties of the 3D structure factor.
"""

import numpy as np
import pytest

from glsf import sf3d

RTOL = 1e-3
ATOL = 1e-3


def sf3d_reference(points, k_vecs):
    """Float64 numpy reference."""
    N = points.shape[0]
    phases = k_vecs.astype(np.float64) @ points.astype(np.float64).T
    c = np.cos(phases).sum(axis=1)
    s = np.sin(phases).sum(axis=1)
    sk = (c**2 + s**2) / N
    k_sq = (k_vecs.astype(np.float64) ** 2).sum(axis=1)
    sk[k_sq < 1e-10] = 0.0
    return sk


def box_kvecs(L, K):
    """Generate k-vectors on a reciprocal box grid 2pi/L * (h,k,l)."""
    f = 2 * np.pi / L
    n = np.arange(-K, K + 1)
    NX, NY, NZ = np.meshgrid(n, n, n, indexing='ij')
    return (np.stack([NX.ravel(), NY.ravel(), NZ.ravel()], axis=1) * f).astype(
        np.float32
    )


def crystal_kvecs(a, K):
    """Generate k-vectors at crystal reciprocal lattice 2pi/a * (h,k,l)."""
    f = 2 * np.pi / a
    n = np.arange(-K, K + 1)
    NX, NY, NZ = np.meshgrid(n, n, n, indexing='ij')
    return (np.stack([NX.ravel(), NY.ravel(), NZ.ravel()], axis=1) * f).astype(
        np.float32
    )


def make_sc(a, replicas):
    """Simple cubic crystal points: a * (i, j, k) for i,j,k in [0, replicas)."""
    n = np.arange(replicas, dtype=np.float32)
    X, Y, Z = np.meshgrid(n, n, n, indexing='ij')
    return a * np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)


def make_bcc(a, replicas):
    """BCC crystal: SC + body-center offset."""
    pts = make_sc(a, replicas)
    body = make_sc(a, replicas) + a / 2
    return np.vstack([pts, body])


def make_fcc(a, replicas):
    """FCC crystal: SC + 3 face-center offsets."""
    pts = make_sc(a, replicas)
    off = a / 2
    face_xy = make_sc(a, replicas) + np.array([off, off, 0])
    face_xz = make_sc(a, replicas) + np.array([off, 0, off])
    face_yz = make_sc(a, replicas) + np.array([0, off, off])
    return np.vstack([pts, face_xy, face_xz, face_yz])


def make_random(L, N, seed=42):
    """Random points uniformly in [-L/2, L/2)^3."""
    rng = np.random.default_rng(seed)
    return rng.uniform(-L / 2, L / 2, (N, 3)).astype(np.float32)


def add_noise(points, sigma, seed=42):
    rng = np.random.default_rng(seed)
    return (points + rng.normal(0, sigma, points.shape)).astype(np.float32)


# ---------------------------------------------------------------------------
# Numerical accuracy
# ---------------------------------------------------------------------------


class TestNumericalAccuracy:
    @pytest.mark.parametrize("name,make_crystal", [
        ("SC", make_sc),
        ("BCC", make_bcc),
        ("FCC", make_fcc),
    ])
    @pytest.mark.parametrize("replicas", [2, 4, 6])
    def test_crystal_accuracy(self, name, make_crystal, replicas):
        a = 1.0
        points = make_crystal(a, replicas)
        L = a * replicas
        k_vecs = box_kvecs(L, K=5)
        sk_gpu = sf3d(points.astype(np.float32), k_vecs)
        sk_ref = sf3d_reference(points, k_vecs)
        np.testing.assert_allclose(sk_gpu, sk_ref, rtol=RTOL, atol=ATOL)

    def test_random_accuracy(self):
        points = make_random(10.0, 1000)
        k_vecs = box_kvecs(10.0, K=3)
        sk_gpu = sf3d(points.astype(np.float32), k_vecs)
        sk_ref = sf3d_reference(points, k_vecs)
        np.testing.assert_allclose(sk_gpu, sk_ref, rtol=RTOL, atol=ATOL)

    def test_noisy_crystal_accuracy(self):
        points = add_noise(make_fcc(1.0, 4), sigma=0.05)
        L = 4.0
        k_vecs = box_kvecs(L, K=5)
        sk_gpu = sf3d(points.astype(np.float32), k_vecs)
        sk_ref = sf3d_reference(points, k_vecs)
        np.testing.assert_allclose(sk_gpu, sk_ref, rtol=RTOL, atol=ATOL)


# ---------------------------------------------------------------------------
# Crystal selection rules: Bragg peaks at the right hkl indices
# ---------------------------------------------------------------------------


class TestCrystalSelectionRules:
    def _get_peaks(self, make_crystal, replicas, K):
        a = 1.0
        points = make_crystal(a, replicas)
        k = crystal_kvecs(a, K)
        sk = sf3d(points.astype(np.float32), k)
        N = len(points)
        f = 2 * np.pi
        peaks = {}
        for i, kv in enumerate(k):
            hkl = np.round(kv / f).astype(int)
            if np.all(hkl == 0):
                continue
            if np.allclose(kv, hkl * f, atol=1e-6):
                peaks[tuple(hkl)] = float(sk[i])
        return N, peaks

    def test_sc_peaks_everywhere(self):
        """SC: all integer hkl should be Bragg peaks with S(k) = N."""
        N, peaks = self._get_peaks(make_sc, 4, K=3)
        for hkl, sk in peaks.items():
            assert sk > N * 0.9, f"SC missing peak at {hkl}: S={sk:.2f}, expected ~{N}"

    def test_bcc_even_sum_only(self):
        """BCC: peaks only where h+k+l is even."""
        N, peaks = self._get_peaks(make_bcc, 4, K=3)
        for (h, k, l), sk in peaks.items():
            if (h + k + l) % 2 == 0:
                assert sk > N * 0.9, f"BCC missing peak at ({h},{k},{l}): S={sk:.2f}"
            else:
                assert sk < 1.0, f"BCC forbidden peak at ({h},{k},{l}): S={sk:.2f}"

    def test_fcc_all_even_or_all_odd(self):
        """FCC: peaks only when h,k,l are all even or all odd."""
        N, peaks = self._get_peaks(make_fcc, 4, K=3)
        for (h, k, l), sk in peaks.items():
            all_even = h % 2 == 0 and k % 2 == 0 and l % 2 == 0
            all_odd = h % 2 == 1 and k % 2 == 1 and l % 2 == 1
            if all_even or all_odd:
                assert sk > N * 0.9, f"FCC missing peak at ({h},{k},{l}): S={sk:.2f}"
            else:
                assert sk < 1.0, f"FCC forbidden peak at ({h},{k},{l}): S={sk:.2f}"


# ---------------------------------------------------------------------------
# Edge cases with known analytical answers
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_single_point(self):
        """Single point: S(k) = 1 for any k."""
        points = np.zeros((1, 3), dtype=np.float32)
        k_vecs = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32)
        sk = sf3d(points, k_vecs)
        np.testing.assert_allclose(sk, [1.0, 1.0, 1.0], atol=1e-6)

    def test_k_zero_masked(self):
        """k=0 must be masked to exactly 0.0."""
        points = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
        k_vecs = np.array([[0, 0, 0], [1, 0, 0]], dtype=np.float32)
        sk = sf3d(points, k_vecs)
        assert sk[0] == 0.0
        assert sk[1] != 0.0

    def test_destructive_interference(self):
        """Two points pi apart along k: phases differ by pi, S(k) = 0."""
        points = np.array([[0, 0, 0], [np.pi, 0, 0]], dtype=np.float32)
        k_vecs = np.array([[1, 0, 0]], dtype=np.float32)
        sk = sf3d(points, k_vecs)
        np.testing.assert_allclose(sk, [0.0], atol=1e-5)

    def test_constructive_interference(self):
        """Two coincident points: S(k) = 2."""
        points = np.array([[1, 0, 0], [1, 0, 0]], dtype=np.float32)
        k_vecs = np.array([[2, 0, 0]], dtype=np.float32)
        sk = sf3d(points, k_vecs)
        np.testing.assert_allclose(sk, [2.0], atol=1e-5)

    def test_perfect_crystal_peak_equals_N(self):
        """For a perfect SC crystal, S(k) should equal N at Bragg peaks."""
        points = make_sc(1.0, 4)
        N = len(points)
        k = crystal_kvecs(1.0, K=2)
        sk = sf3d(points.astype(np.float32), k)
        f = 2 * np.pi
        for i, kv in enumerate(k):
            hkl = np.round(kv / f).astype(int)
            if np.all(hkl == 0):
                continue
            if np.allclose(kv, hkl * f, atol=1e-6):
                np.testing.assert_allclose(
                    sk[i], N, rtol=1e-3, err_msg=f"Peak at {hkl}: got {sk[i]:.2f}, expected {N}"
                )


# ---------------------------------------------------------------------------
# Statistical properties of random systems
# ---------------------------------------------------------------------------


class TestStatisticalProperties:
    def test_random_mean_near_one(self):
        """For N>>1 random points, <S(k)> -> 1 for k != 0."""
        points = make_random(10.0, 5000)
        k_vecs = box_kvecs(10.0, K=5)
        sk = sf3d(points.astype(np.float32), k_vecs)
        nonzero = sk != 0.0
        mean_sk = np.mean(sk[nonzero])
        assert abs(mean_sk - 1.0) < 0.1, f"Mean S(k) = {mean_sk:.4f}, expected ~1.0"

    def test_variance_decreases_with_N(self):
        """Var(S(k)) for random systems ~ 2/N, so larger N gives smaller variance."""
        vars_ = {}
        for N in [500, 2000]:
            points = make_random(10.0, N)
            k_vecs = box_kvecs(10.0, K=3)
            sk = sf3d(points.astype(np.float32), k_vecs)
            nonzero = sk != 0.0
            vars_[N] = np.var(sk[nonzero])
        assert vars_[2000] < vars_[500], (
            f"Var(N=2000)={vars_[2000]:.4f} should be < Var(N=500)={vars_[500]:.4f}"
        )
