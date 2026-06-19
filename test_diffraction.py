"""Tests for wgpu_diffraction against a float64 numpy reference.

Uses freud.data to generate physically meaningful test systems (SC, BCC, FCC
crystals, random points) and validates both numerical accuracy and
crystallographic selection rules.
"""
# ruff: noqa: E741

import numpy as np
import pytest
import freud.data

from wgpu_diffraction import sf3d

RTOL = 5e-5
ATOL = 5e-6


def direct_ft_numpy(points, k_vecs):
    """Reference implementation for a three-dimensional structure factor."""
    N = points.shape[0]
    phases = k_vecs.astype(np.float64) @ points.astype(np.float64).T
    c = np.cos(phases).sum(axis=1)
    s = np.sin(phases).sum(axis=1)
    return ((c**2 + s**2) / N).astype(np.float64)


def crystal_kvecs(L, K):
    """Generate k-vectors at reciprocal lattice points 2pi/L * (h,k,l)."""
    f = 2 * np.pi / L
    n = np.arange(-K, K + 1)
    NX, NY, NZ = np.meshgrid(n, n, n, indexing="ij")
    return (np.stack([NX.ravel(), NY.ravel(), NZ.ravel()], axis=1) * f).astype(
        np.float32
    )


def bragg_indices_L1(k):
    """Indices of k-vectors at reciprocal lattice points of a unit (L=1) crystal.

    Hardcodes the 2π spacing — callers must pass k from `crystal_kvecs(1.0, K)`.
    """
    f = 2 * np.pi
    hkl = np.round(k / f).astype(int)
    on_bragg = np.all(np.abs(k - hkl * f) < 1e-6, axis=1)
    nonzero = np.any(hkl != 0, axis=1)
    return np.where(on_bragg & nonzero)[0]


class TestNumericalAccuracy:
    @pytest.mark.parametrize(
        "name,uc_func",
        [
            ("SC", freud.data.UnitCell.sc),
            ("BCC", freud.data.UnitCell.bcc),
            ("FCC", freud.data.UnitCell.fcc),
        ],
    )
    @pytest.mark.parametrize("replicas", [2, 4, 6])
    def test_crystal_accuracy(self, name, uc_func, replicas):
        box, points = uc_func().generate_system(replicas)
        k_vecs = crystal_kvecs(box.Lx, K=5)
        sk_gpu = sf3d(points.astype(np.float32), k_vecs)
        sk_ref = direct_ft_numpy(points, k_vecs)
        np.testing.assert_allclose(sk_gpu, sk_ref, rtol=RTOL, atol=ATOL)

    @pytest.mark.parametrize("N", [500, 1000, 2000])
    def test_random_accuracy(self, N):
        _, points = freud.data.make_random_system(10.0, N, seed=42)
        k_vecs = crystal_kvecs(10.0, K=3)
        sk_gpu = sf3d(points.astype(np.float32), k_vecs)
        sk_ref = direct_ft_numpy(points, k_vecs)
        np.testing.assert_allclose(sk_gpu, sk_ref, rtol=RTOL, atol=ATOL)


class TestCrystalSelectionRules:
    """Validate Bragg peaks are at the correct HKL indices.

    Forbidden-peak tolerance (RTOL = 5e-5) is tight against float32 cancellation
    residual, which scales as ~N·eps. Tests pin replicas=4 (N=64) — bumping
    replicas will require loosening the forbidden bound proportionally.
    """

    def _get_peaks(self, uc_func, replicas, K, is_allowed):
        _, points = uc_func().generate_system(replicas)
        k = crystal_kvecs(1.0, K)
        sk = sf3d(points.astype(np.float32), k)
        N = len(points)
        f = 2 * np.pi
        idx = bragg_indices_L1(k)
        hkl = np.round(k[idx] / f).astype(int)
        allowed_mask = np.array([is_allowed(tuple(h)) for h in hkl])
        return N, sk[idx][allowed_mask], sk[idx][~allowed_mask]

    @pytest.mark.parametrize(
        "name,uc_func,is_allowed",
        [
            ("SC", freud.data.UnitCell.sc, lambda _: True),
            ("BCC", freud.data.UnitCell.bcc, lambda hkl: sum(hkl) % 2 == 0),
            (
                "FCC",
                freud.data.UnitCell.fcc,
                lambda hkl: (
                    all(v % 2 == 0 for v in hkl) or all(v % 2 == 1 for v in hkl)
                ),
            ),
        ],
    )
    def test_selection_rules(self, name, uc_func, is_allowed):
        """Allowed Bragg peaks have S(k) ~ N; forbidden peaks have S(k) ~ 0."""
        N, allowed, forbidden = self._get_peaks(uc_func, 4, K=3, is_allowed=is_allowed)
        np.testing.assert_array_less(
            N * (1.0 - RTOL),
            allowed,
            err_msg=f"{name}: some allowed peaks below {N * (1.0 - RTOL):.2f}",
        )
        if len(forbidden) > 0:
            np.testing.assert_array_less(
                forbidden,
                RTOL,
                err_msg=f"{name}: some forbidden peaks above {RTOL}",
            )


class TestEdgeCases:
    """Check edge cases with known analytical answers."""

    def test_single_point(self):
        """Single point: S(k) = 1 for any k."""
        points = np.zeros((1, 3), dtype=np.float32)
        k_vecs = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32)
        sk = sf3d(points, k_vecs)
        np.testing.assert_allclose(sk, [1.0, 1.0, 1.0], atol=ATOL)

    def test_k_zero_equals_N(self):
        """k=0: S(0) = N (all particles in phase)."""
        points = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
        k_vecs = np.array([[0, 0, 0]], dtype=np.float32)
        sk = sf3d(points, k_vecs)
        np.testing.assert_allclose(sk[0], 2.0, atol=ATOL)

    def test_destructive_interference(self):
        """Two points pi apart along k: phases differ by pi, S(k) = 0."""
        points = np.array([[0, 0, 0], [np.pi, 0, 0]], dtype=np.float32)
        k_vecs = np.array([[1, 0, 0]], dtype=np.float32)
        sk = sf3d(points, k_vecs)
        np.testing.assert_allclose(sk, [0.0], atol=ATOL)

    def test_constructive_interference(self):
        """Two coincident points: S(k) = 2."""
        points = np.array([[1, 0, 0], [1, 0, 0]], dtype=np.float32)
        k_vecs = np.array([[2, 0, 0]], dtype=np.float32)
        sk = sf3d(points, k_vecs)
        np.testing.assert_allclose(sk, [2.0], atol=ATOL)


class TestDebyeWaller:
    """Debye-Waller factor validation for crystals with Gaussian noise.

    For positions r_i = R_i + δ_i with independent δ_i ~ N(0, σ²I),
    the ensemble-averaged structure factor has a closed form:
        <S(k)> = 1 + exp(-σ²|k|²) × (S₀(k) - 1)
    where S₀(k) is the perfect-crystal structure factor.
    """

    @pytest.mark.parametrize("sigma", [0.01, 0.05, 0.1, 0.2])
    def test_bragg_peak_attenuation(self, sigma):
        """Bragg peak intensity is damped by exp(-σ²|k|²) (Debye-Waller factor)."""
        _, perfect_pts = freud.data.UnitCell.sc().generate_system(3)
        k = crystal_kvecs(1.0, K=3)
        bragg_idx = bragg_indices_L1(k)

        s0 = sf3d(perfect_pts.astype(np.float32), k)
        s0_bragg = s0[bragg_idx]
        k_bragg = k[bragg_idx]
        k_sq_bragg = np.sum(k_bragg.astype(np.float64) ** 2, axis=1)

        dwf = np.exp(-(sigma**2) * k_sq_bragg)
        expected = 1.0 + dwf * (s0_bragg.astype(np.float64) - 1.0)

        testable = expected > 2.0
        if not np.any(testable):
            pytest.skip(f"σ={sigma}: no Bragg peaks with measurable DWF")

        n_samples = 100
        rng = np.random.default_rng(42)
        sk_sum = np.zeros(len(bragg_idx))
        for _ in range(n_samples):
            noise = rng.normal(0, sigma, perfect_pts.shape).astype(np.float32)
            noisy_pts = perfect_pts.astype(np.float32) + noise
            sk = sf3d(noisy_pts, k)
            sk_sum += sk[bragg_idx]
        sk_mean = sk_sum / n_samples

        np.testing.assert_allclose(
            sk_mean[testable],
            expected[testable],
            rtol=0.12,
            err_msg=f"σ={sigma}: Bragg peak DWF mismatch",
        )


class TestStatisticalProperties:
    """Diffraction of random systems have some properties"""

    def test_random_mean_near_one(self):
        """For N>>1 random points, <S(k)> -> 1 for k != 0."""
        _, points = freud.data.make_random_system(10.0, 5000, seed=42)
        k_vecs = crystal_kvecs(10.0, K=5)
        sk = sf3d(points.astype(np.float32), k_vecs)
        k_sq = np.sum(k_vecs.astype(np.float64) ** 2, axis=1)
        nonzero = k_sq > 1e-10
        mean_sk = np.mean(sk[nonzero])
        np.testing.assert_allclose(
            mean_sk, 1.0, atol=0.05, err_msg=f"Mean S(k) = {mean_sk:.4f}, expected ~1.0"
        )


class TestDebyeValidation:
    """Validate against the Debye scattering equation (independent algorithm).

    The Debye formula computes the isotropic (spherically-averaged) structure factor
    from pairwise distances: S(q) = (1/N) Σ_i Σ_j sinc(q |r_ij|), where sinc is
    the unnormalized sin(x)/x. This is an O(N²) algorithm completely independent
    of the direct Fourier approach.
    """

    @staticmethod
    def _debye_ssf(points, q_values):
        """Compute S(q) via the Debye scattering equation (float64 reference)."""
        pts = points.astype(np.float64)
        N = len(pts)
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        distances = np.sqrt(np.sum(diff**2, axis=2)).ravel()
        S = np.zeros(len(q_values))
        for i, q in enumerate(q_values):
            qd = q * distances
            # np.sinc(x) = sin(πx)/(πx), so np.sinc(qd/π) = sin(qd)/qd
            S[i] = np.sum(np.sinc(qd / np.pi)) / N
        return S

    def test_debye_matches_direct(self):
        """Compare GPU direct FT against Debye scattering on an FCC crystal."""
        _, points = freud.data.UnitCell.fcc().generate_system(
            3, sigma_noise=0.05, seed=42
        )

        q_values = np.array([2.0, 4.0, 6.0, 8.0, 10.0])
        rng = np.random.default_rng(42)
        n_per_shell = 2000
        k_vecs = []
        for q in q_values:
            dirs = rng.standard_normal((n_per_shell, 3)).astype(np.float32)
            dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
            k_vecs.append(dirs * q)
        k_vecs = np.vstack(k_vecs).astype(np.float32)

        sk_gpu = sf3d(points.astype(np.float32), k_vecs)

        for i, q in enumerate(q_values):
            shell = sk_gpu[i * n_per_shell : (i + 1) * n_per_shell]
            sk_shell_mean = np.mean(shell)
            sk_debye = self._debye_ssf(points, np.array([q]))[0]
            np.testing.assert_allclose(
                sk_shell_mean,
                sk_debye,
                rtol=0.05,
                err_msg=f"q={q}: GPU mean={sk_shell_mean:.2f}, Debye={sk_debye:.2f}",
            )


class TestVacancyDisorder:
    """Structure factor with random vacancies (quenched disorder).

    For a crystal with M lattice sites where each site is independently
    occupied with probability p = 1-c, the disorder-averaged structure factor is:
        <S(k)> = c + (1-c) S_full(k)
    where S_full(k) is the perfect-crystal result (normalized by M).
    """

    @staticmethod
    def _introduce_vacancies(points, vacancy_frac, rng):
        """Remove a random fraction of points to simulate vacancies."""
        n_remove = int(len(points) * vacancy_frac)
        indices = rng.choice(len(points), size=n_remove, replace=False)
        mask = np.ones(len(points), dtype=bool)
        mask[indices] = False
        return points[mask]

    @pytest.mark.parametrize("c", [0.05, 0.15])
    def test_bragg_peak_reduction(self, c):
        """Bragg peaks follow <S(k)> = c + (1-c) S_full(k)."""
        _, perfect_pts = freud.data.UnitCell.sc().generate_system(4)
        k = crystal_kvecs(1.0, K=3)
        bragg_idx = bragg_indices_L1(k)

        s_full = sf3d(perfect_pts.astype(np.float32), k)
        expected = c + (1 - c) * s_full[bragg_idx].astype(np.float64)

        n_samples = 100
        rng = np.random.default_rng(42)
        sk_sum = np.zeros(len(bragg_idx))
        for _ in range(n_samples):
            pts = self._introduce_vacancies(perfect_pts, c, rng)
            sk_sum += sf3d(pts.astype(np.float32), k)[bragg_idx]
        sk_mean = sk_sum / n_samples

        np.testing.assert_allclose(
            sk_mean,
            expected,
            rtol=0.12,
            err_msg=f"c={c}: vacancy Bragg peak mismatch",
        )


class TestLauePeakProfile:
    """Finite-size peak broadening follows the Laue function.

    For an MxMxM SC crystal (a=1), near Bragg peak G = 2π(h₀,k₀,l₀) along x:
        S(G + δ x̂) = (1/M) sin²(Mδ/2) / sin²(δ/2)
    At exact Bragg peak (δ=0): S = M³ = N.
    First zeros at δ = ±2π/M.
    """

    @staticmethod
    def _laue_1d(delta, M):
        """Laue function: sin²(Mδ/2) / (M sin²(δ/2)), with limit M at δ=0."""
        delta = np.asarray(delta, dtype=np.float64)
        result = np.empty_like(delta)
        near_zero = np.abs(delta) < 1e-10
        far = ~near_zero
        result[near_zero] = M
        if np.any(far):
            hd = delta[far] / 2
            result[far] = np.sin(M * hd) ** 2 / (M * np.sin(hd) ** 2)
        return result

    @pytest.mark.parametrize("M", [4, 6])
    def test_peak_profile_through_bragg(self, M):
        """Peak shape matches the Laue function through (2π,0,0)."""
        _, points = freud.data.UnitCell.sc().generate_system(M)
        f = 2 * np.pi

        delta = np.linspace(-np.pi, np.pi, 500, dtype=np.float32)
        k_vecs = np.zeros((len(delta), 3), dtype=np.float32)
        k_vecs[:, 0] = f + delta

        sk = sf3d(points.astype(np.float32), k_vecs)

        # Analytical: x-profile × M² (y,z still at Bragg peak)
        expected = self._laue_1d(delta.astype(np.float64), M) * M**2

        np.testing.assert_allclose(
            sk.astype(np.float64),
            expected,
            rtol=5e-4,
            atol=5e-5,
            err_msg=f"M={M}: Laue profile mismatch",
        )
