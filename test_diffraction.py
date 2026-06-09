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
NEAR_ONE = 1.0 - RTOL
NEAR_ZERO = RTOL


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

    @pytest.mark.parametrize(
        "name,uc_func",
        [
            ("SC", freud.data.UnitCell.sc),
            ("BCC", freud.data.UnitCell.bcc),
            ("FCC", freud.data.UnitCell.fcc),
        ],
    )
    def test_noisy_crystal_accuracy(self, name, uc_func):
        box, points = uc_func().generate_system(4, sigma_noise=0.05, seed=42)
        k_vecs = crystal_kvecs(box.Lx, K=5)
        sk_gpu = sf3d(points.astype(np.float32), k_vecs)
        sk_ref = direct_ft_numpy(points, k_vecs)
        np.testing.assert_allclose(sk_gpu, sk_ref, rtol=RTOL, atol=ATOL)


class TestCrystalSelectionRules:
    """Validate Bragg peaks are at the correct HKL indices."""

    def _get_peaks(self, uc_func, replicas, K, is_allowed):
        _, points = uc_func().generate_system(replicas)
        k = crystal_kvecs(1.0, K)
        sk = sf3d(points.astype(np.float32), k)
        N = len(points)
        f = 2 * np.pi
        allowed, forbidden = [], []
        for i, kv in enumerate(k):
            hkl = tuple(np.round(kv / f).astype(int))
            if hkl == (0, 0, 0):
                continue
            if np.allclose(kv, np.array(hkl) * f, atol=1e-6):
                if is_allowed(hkl):
                    allowed.append(sk[i])
                else:
                    forbidden.append(sk[i])
        return N, np.array(allowed), np.array(forbidden)

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
            N * NEAR_ONE,
            allowed,
            err_msg=f"{name}: some allowed peaks below {N * NEAR_ONE:.2f}",
        )
        if len(forbidden) > 0:
            np.testing.assert_array_less(
                forbidden,
                NEAR_ZERO,
                err_msg=f"{name}: some forbidden peaks above {NEAR_ZERO}",
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
        k_vecs = np.array([[0, 0, 0], [1, 0, 0]], dtype=np.float32)
        sk = sf3d(points, k_vecs)
        np.testing.assert_allclose(sk[0], 2.0, atol=ATOL)
        assert sk[1] != 0.0

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

    def test_perfect_crystal_peak_equals_N(self):
        """For a perfect SC crystal, S(k) should equal N at Bragg peaks."""
        _, points = freud.data.UnitCell.sc().generate_system(4)
        N = len(points)
        k = crystal_kvecs(1.0, K=2)
        sk = sf3d(points.astype(np.float32), k)
        f = 2 * np.pi
        bragg = [
            sk[i]
            for i, kv in enumerate(k)
            if not np.all(np.round(kv / f).astype(int) == 0)
            and np.allclose(kv, np.round(kv / f).astype(int) * f, atol=1e-6)
        ]
        np.testing.assert_allclose(
            bragg,
            N,
            rtol=RTOL,
            err_msg=f"Bragg peaks should equal N={N}",
        )


_DIAMOND_UC = freud.data.UnitCell(
    freud.Box.cube(1.0),
    np.array(
        [
            [0, 0, 0],
            [0.5, 0.5, 0],
            [0, 0.5, 0.5],
            [0.5, 0, 0.5],
            [0.25, 0.25, 0.25],
            [0.75, 0.75, 0.25],
            [0.25, 0.75, 0.75],
            [0.75, 0.25, 0.75],
        ]
    ),
)


class TestDiamondStructureFactor:
    """Diamond cubic selection rules (see Wikipedia: Structure factor § Examples).

    8 atoms per cubic cell. For identical particles (f=1):
      h+k+l = 4N     → S(k) = N   (all 8 basis atoms in phase)
      h+k+l = 2N+1   → S(k) = N/2 (partial cancellation)
      h+k+l = 4N+2   → S(k) = 0   (total cancellation)
      mixed parity    → S(k) = 0   (FCC sublattice extinction)
    """

    def _get_diamond_peaks(self, replicas, K):
        _, points = _DIAMOND_UC.generate_system(replicas)
        N = len(points)
        k = crystal_kvecs(1.0, K)
        sk = sf3d(points.astype(np.float32), k)

        f = 2 * np.pi
        hkl = np.round(k / f).astype(int)
        on_bragg = np.all(np.abs(k - hkl * f) < 1e-6, axis=1)
        nonzero = np.any(hkl != 0, axis=1)
        mask = on_bragg & nonzero

        h, k_, l = hkl[mask, 0], hkl[mask, 1], hkl[mask, 2]
        s = sk[mask]
        parity_uniform = (h % 2 == k_ % 2) & (k_ % 2 == l % 2)
        hkl_sum = h + k_ + l

        return (
            N,
            s[parity_uniform & (hkl_sum % 4 == 0)],
            s[parity_uniform & (hkl_sum % 2 == 1)],
            s[parity_uniform & (hkl_sum % 4 == 2)],
            s[~parity_uniform],
        )

    def test_mixed_parity_extinct(self):
        """Mixed parity hkl are extinct (FCC sublattice extinction)."""
        _, _, _, _, mixed = self._get_diamond_peaks(4, K=3)
        np.testing.assert_array_less(
            mixed,
            NEAR_ZERO,
            err_msg="Diamond: mixed-parity peaks should be extinct",
        )

    def test_mod4_equals_0_gives_full_intensity(self):
        """h+k+l ≡ 0 mod 4 → S(k) = N (all 8 atoms in phase)."""
        N, mod4_0, _, _, _ = self._get_diamond_peaks(4, K=3)
        np.testing.assert_allclose(
            mod4_0,
            N,
            rtol=RTOL,
            err_msg=f"Diamond: h+k+l=4N peaks should equal N={N}",
        )

    def test_odd_sum_gives_half_intensity(self):
        """h+k+l odd → S(k) = N/2 (partial cancellation)."""
        N, _, odd, _, _ = self._get_diamond_peaks(4, K=3)
        np.testing.assert_allclose(
            odd,
            N / 2,
            rtol=RTOL,
            err_msg=f"Diamond: h+k+l odd peaks should equal N/2={N / 2}",
        )

    def test_mod4_equals_2_extinct(self):
        """h+k+l ≡ 2 mod 4 → S(k) = 0 (total cancellation)."""
        _, _, _, mod4_2, _ = self._get_diamond_peaks(4, K=3)
        np.testing.assert_array_less(
            mod4_2,
            NEAR_ZERO,
            err_msg="Diamond: h+k+l=4N+2 peaks should be extinct",
        )


class TestDebyeWaller:
    """Debye-Waller factor validation for crystals with Gaussian noise.

    For positions r_i = R_i + δ_i with independent δ_i ~ N(0, σ²I),
    the ensemble-averaged structure factor has a closed form:
        <S(k)> = 1 + exp(-σ²|k|²) × (S₀(k) - 1)
    where S₀(k) is the perfect-crystal structure factor.
    """

    @staticmethod
    def _bragg_indices(k, K_max):
        """Return indices of k-vectors that lie on the reciprocal lattice."""
        f = 2 * np.pi
        hkl = np.round(k / f).astype(int)
        on_bragg = np.all(np.abs(k - hkl * f) < 1e-6, axis=1)
        nonzero = np.any(hkl != 0, axis=1)
        return np.where(on_bragg & nonzero)[0]

    @pytest.mark.parametrize("sigma", [0.01, 0.05, 0.1, 0.2])
    def test_bragg_peak_attenuation(self, sigma):
        """Bragg peak intensity is damped by exp(-σ²|k|²) (Debye-Waller factor)."""
        _, perfect_pts = freud.data.UnitCell.sc().generate_system(3)
        k = crystal_kvecs(1.0, K=3)
        bragg_idx = self._bragg_indices(k, K_max=3)

        # Compute perfect-crystal S₀(k) at Bragg peaks
        s0 = sf3d(perfect_pts.astype(np.float32), k)
        s0_bragg = s0[bragg_idx]
        k_bragg = k[bragg_idx]
        k_sq_bragg = np.sum(k_bragg.astype(np.float64) ** 2, axis=1)

        # Expected: <S(k)> = 1 + exp(-σ²k²) × (S₀(k) - 1)
        dwf = np.exp(-(sigma**2) * k_sq_bragg)
        expected = 1.0 + dwf * (s0_bragg.astype(np.float64) - 1.0)

        # Only test peaks where the signal is still above the diffuse background (~1)
        testable = expected > 2.0
        if not np.any(testable):
            pytest.skip(f"σ={sigma}: no Bragg peaks with measurable DWF")

        # Average over noisy realizations
        n_samples = 300
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
            rtol=0.10,
            err_msg=f"σ={sigma}: Bragg peak DWF mismatch",
        )

    def test_diffuse_background_at_bragg(self):
        """Intensity lost from Bragg peaks appears as diffuse scattering.

        The diffuse background at Bragg peaks is 1 - exp(-σ²|k|²).
        As σ increases, more intensity moves from Bragg peaks to background.
        """
        _, perfect_pts = freud.data.UnitCell.sc().generate_system(3)
        k = crystal_kvecs(1.0, K=2)
        bragg_idx = self._bragg_indices(k, K_max=2)

        # Perfect crystal S₀ = N at all Bragg peaks
        # With noise: <S> = 1 + exp(-σ²k²)(N - 1)
        # Diffuse fraction = 1 - exp(-σ²k²)
        # Verify: higher σ → lower Bragg intensity
        sigma_low, sigma_high = 0.02, 0.15
        n_samples = 50
        rng = np.random.default_rng(7)

        def avg_bragg(sigma):
            sk_sum = np.zeros(len(bragg_idx))
            for _ in range(n_samples):
                noise = rng.normal(0, sigma, perfect_pts.shape).astype(np.float32)
                noisy_pts = perfect_pts.astype(np.float32) + noise
                sk_sum += sf3d(noisy_pts, k)[bragg_idx]
            return np.mean(sk_sum) / n_samples

        mean_low = avg_bragg(sigma_low)
        mean_high = avg_bragg(sigma_high)
        np.testing.assert_array_less(
            mean_high,
            mean_low,
            err_msg=f"Higher σ should reduce Bragg intensity: σ={sigma_high} mean={mean_high:.2f} >= σ={sigma_low} mean={mean_low:.2f}",
        )

    def test_perfect_crystal_dwf_is_one(self):
        """With σ=0 the Debye-Waller factor is exactly 1 (no attenuation)."""
        _, perfect_pts = freud.data.UnitCell.sc().generate_system(3)
        k = crystal_kvecs(1.0, K=2)
        bragg_idx = self._bragg_indices(k, K_max=2)
        N = len(perfect_pts)

        sk = sf3d(perfect_pts.astype(np.float32), k)
        np.testing.assert_allclose(
            sk[bragg_idx],
            N,
            rtol=RTOL,
            err_msg="Perfect crystal Bragg peaks should equal N (DWF=1)",
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

    def test_variance_decreases_with_N(self):
        """Var(S(k)) for random systems ~ 2/N, so larger N gives smaller variance."""
        vars_ = {}
        for N in [500, 2000]:
            box, points = freud.data.make_random_system(10.0, N, seed=42)
            k_vecs = crystal_kvecs(10.0, K=3)
            sk = sf3d(points.astype(np.float32), k_vecs)
            k_sq = np.sum(k_vecs.astype(np.float64) ** 2, axis=1)
            nonzero = k_sq > 1e-10
            vars_[N] = np.var(sk[nonzero])
        np.testing.assert_array_less(
            vars_[2000],
            vars_[500],
            err_msg=f"Var(N=2000)={vars_[2000]:.4f} should be < Var(N=500)={vars_[500]:.4f}",
        )

    def test_large_k_scattering_goes_to_one(self):
        """At large |k|, scattering becomes incoherent and <S(k)> -> 1."""
        _, points = freud.data.make_random_system(10.0, 1000, seed=1)
        # Generate k-vectors with large |k| on a spherical shell
        rng = np.random.default_rng(1)
        directions = rng.standard_normal((5000, 3)).astype(np.float32)
        directions /= np.linalg.norm(directions, axis=1, keepdims=True)
        k_magnitude = 200.0  # much larger than 2*pi/L
        k_vecs = directions * k_magnitude
        sk = sf3d(points, k_vecs)
        np.testing.assert_allclose(
            np.mean(sk),
            1.0,
            atol=0.05,
            err_msg=f"Large-k mean S(k) = {np.mean(sk):.4f}, expected ~1.0",
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
        # Pairwise distance matrix
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        distances = np.sqrt(np.sum(diff**2, axis=2)).ravel()
        S = np.zeros(len(q_values))
        for i, q in enumerate(q_values):
            qd = q * distances
            # sinc(x/π) = sin(x)/x, but we want sin(qd)/(qd) so use np.sinc(qd/π)
            S[i] = np.sum(np.sinc(qd / np.pi)) / N
        return S

    def test_debye_matches_direct(self):
        """Compare GPU direct FT against Debye scattering on an FCC crystal."""
        box, points = freud.data.UnitCell.fcc().generate_system(
            3, sigma_noise=0.05, seed=42
        )

        # Choose q values and generate k-vectors on spherical shells
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

        # Average GPU result per shell and compare to Debye
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

    @pytest.mark.parametrize(
        "name,uc_func",
        [
            ("SC", freud.data.UnitCell.sc),
            ("BCC", freud.data.UnitCell.bcc),
            ("FCC", freud.data.UnitCell.fcc),
        ],
    )
    def test_debye_crystal_types(self, name, uc_func):
        """Debye validation on multiple crystal types with noise."""
        box, points = uc_func().generate_system(2, sigma_noise=0.1, seed=7)

        q_values = np.array([3.0, 7.0, 12.0])
        rng = np.random.default_rng(7)
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
            sk_debye = self._debye_ssf(points, np.array([q]))[0]
            np.testing.assert_allclose(
                np.mean(shell),
                sk_debye,
                rtol=0.05,
                err_msg=f"{name} q={q}: GPU mean={np.mean(shell):.2f}, Debye={sk_debye:.2f}",
            )

    def test_debye_random_system(self):
        """Debye validation on a random (isotropic) system."""
        _, points = freud.data.make_random_system(10.0, 200, seed=99)

        q_values = np.array([1.0, 3.0, 5.0, 8.0])
        rng = np.random.default_rng(99)
        n_per_shell = 4000
        k_vecs = []
        for q in q_values:
            dirs = rng.standard_normal((n_per_shell, 3)).astype(np.float32)
            dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
            k_vecs.append(dirs * q)
        k_vecs = np.vstack(k_vecs).astype(np.float32)

        sk_gpu = sf3d(points.astype(np.float32), k_vecs)
        sk_debye = self._debye_ssf(points, q_values)

        for i, q in enumerate(q_values):
            shell = sk_gpu[i * n_per_shell : (i + 1) * n_per_shell]
            np.testing.assert_allclose(
                np.mean(shell),
                sk_debye[i],
                rtol=0.05,
                err_msg=f"Random q={q}: GPU mean={np.mean(shell):.2f}, Debye={sk_debye[i]:.2f}",
            )

    def test_s0_equals_N(self):
        """S(0) = N for any configuration (fundamental normalization).

        At k=0, all phases are 0, so S(0) = (1/N)|sum(1)|^2 = N.
        """
        _, points = freud.data.make_random_system(10.0, 500, seed=1)
        N = len(points)
        k_vecs = np.array([[0, 0, 0]], dtype=np.float32)
        sk = sf3d(points.astype(np.float32), k_vecs)
        np.testing.assert_allclose(
            sk[0],
            N,
            rtol=1e-3,
            err_msg=f"S(0) = {sk[0]:.2f}, expected N={N}",
        )

    def test_output_non_negative(self):
        """S(k) is always >= 0 (sum of squared moduli / N)."""
        _, points = freud.data.make_random_system(10.0, 500, seed=1)
        k_vecs = crystal_kvecs(10.0, K=5)
        sk = sf3d(points.astype(np.float32), k_vecs)
        np.testing.assert_array_less(
            -ATOL,
            sk,
            err_msg="S(k) must be non-negative",
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

    @staticmethod
    def _bragg_indices(k):
        f = 2 * np.pi
        hkl = np.round(k / f).astype(int)
        on_bragg = np.all(np.abs(k - hkl * f) < 1e-6, axis=1)
        nonzero = np.any(hkl != 0, axis=1)
        return np.where(on_bragg & nonzero)[0]

    @pytest.mark.parametrize("c", [0.05, 0.15])
    def test_bragg_peak_reduction(self, c):
        """Bragg peaks follow <S(k)> = c + (1-c) S_full(k)."""
        _, perfect_pts = freud.data.UnitCell.sc().generate_system(4)
        k = crystal_kvecs(1.0, K=3)
        bragg_idx = self._bragg_indices(k)

        s_full = sf3d(perfect_pts.astype(np.float32), k)
        expected = c + (1 - c) * s_full[bragg_idx].astype(np.float64)

        n_samples = 300
        rng = np.random.default_rng(42)
        sk_sum = np.zeros(len(bragg_idx))
        for _ in range(n_samples):
            pts = self._introduce_vacancies(perfect_pts, c, rng)
            sk_sum += sf3d(pts.astype(np.float32), k)[bragg_idx]
        sk_mean = sk_sum / n_samples

        np.testing.assert_allclose(
            sk_mean,
            expected,
            rtol=0.10,
            err_msg=f"c={c}: vacancy Bragg peak mismatch",
        )

    @pytest.mark.parametrize("c", [0.05, 0.15])
    def test_diffuse_background(self, c):
        """Off-Bragg diffuse scattering ~ c from vacancy disorder."""
        _, perfect_pts = freud.data.UnitCell.sc().generate_system(4)

        # k-vectors at half-integer hkl — off the reciprocal lattice
        f = 2 * np.pi
        n = np.arange(-3, 4)
        H, K_, L = np.meshgrid(n, n, n, indexing="ij")
        k = np.stack([H.ravel(), K_.ravel(), L.ravel()], axis=1).astype(np.float32)
        k = k * f + np.array([f / 2, 0, 0], dtype=np.float32)
        nonzero = np.any(k != 0, axis=1)
        k = k[nonzero]

        n_samples = 300
        rng = np.random.default_rng(99)
        sk_sum = np.zeros(len(k))
        for _ in range(n_samples):
            pts = self._introduce_vacancies(perfect_pts, c, rng)
            sk_sum += sf3d(pts.astype(np.float32), k)
        sk_mean = sk_sum / n_samples

        np.testing.assert_allclose(
            np.mean(sk_mean),
            c,
            atol=0.05,
            err_msg=f"c={c}: off-Bragg <S>={np.mean(sk_mean):.4f}, expected ~{c}",
        )

    def test_monotonic_with_concentration(self):
        """Higher vacancy fraction → lower Bragg intensity."""
        _, perfect_pts = freud.data.UnitCell.sc().generate_system(4)
        k = crystal_kvecs(1.0, K=2)
        bragg_idx = self._bragg_indices(k)

        n_samples = 50
        rng = np.random.default_rng(7)

        def avg_bragg(c):
            sk_sum = np.zeros(len(bragg_idx))
            for _ in range(n_samples):
                pts = self._introduce_vacancies(perfect_pts, c, rng)
                sk_sum += sf3d(pts.astype(np.float32), k)[bragg_idx]
            return np.mean(sk_sum) / n_samples

        mean_low = avg_bragg(0.02)
        mean_high = avg_bragg(0.20)
        np.testing.assert_array_less(
            mean_high,
            mean_low,
            err_msg=f"Higher c should reduce Bragg: c=0.20 mean={mean_high:.2f} >= c=0.02 mean={mean_low:.2f}",
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

    def test_peak_width_scales_with_M(self):
        """Larger crystal → narrower peaks (FWHM ~ 1/M)."""
        fwhms = {}
        for M in [4, 8]:
            _, points = freud.data.UnitCell.sc().generate_system(M)
            f = 2 * np.pi
            delta = np.linspace(-np.pi, np.pi, 1000, dtype=np.float32)
            k_vecs = np.zeros((len(delta), 3), dtype=np.float32)
            k_vecs[:, 0] = f + delta
            sk = sf3d(points.astype(np.float32), k_vecs)

            half_max = sk.max() / 2
            above = np.where(sk >= half_max)[0]
            fwhms[M] = delta[above[-1]] - delta[above[0]]

        np.testing.assert_array_less(
            fwhms[8] * 1.5,
            fwhms[4],
            err_msg=f"FWHM should scale ~1/M: M=4 fwhm={fwhms[4]:.4f}, M=8 fwhm={fwhms[8]:.4f}",
        )


def orthorhombic_kvecs(ax, ay, az, K):
    """Generate k-vectors at (2πh/ax, 2πk/ay, 2πl/az) for integer h,k,l."""
    n = np.arange(-K, K + 1)
    H, K_, L = np.meshgrid(n, n, n, indexing="ij")
    kvecs = np.stack([H.ravel(), K_.ravel(), L.ravel()], axis=1).astype(np.float64)
    kvecs[:, 0] *= 2 * np.pi / ax
    kvecs[:, 1] *= 2 * np.pi / ay
    kvecs[:, 2] *= 2 * np.pi / az
    return kvecs.astype(np.float32)


class TestUniformStrain:
    """Affine strain shifts Bragg peaks via G' = (I+ε)⁻ᵀ G.

    For diagonal strain ε = diag(ε_x, ε_y, ε_z), the new reciprocal lattice
    has spacings 2π/(a(1+ε_i)) along each axis.
    """

    @staticmethod
    def _apply_strain(points, strain_diag):
        """Apply diagonal affine strain: r → diag(1+ε) r."""
        scale = (1 + np.asarray(strain_diag, dtype=np.float64)).astype(np.float32)
        return points * scale

    @staticmethod
    def _bragg_indices(k):
        f = 2 * np.pi
        hkl = np.round(k / f).astype(int)
        on_bragg = np.all(np.abs(k - hkl * f) < 1e-6, axis=1)
        nonzero = np.any(hkl != 0, axis=1)
        return np.where(on_bragg & nonzero)[0]

    @pytest.mark.parametrize("eps", [0.03, 0.05, 0.08])
    def test_hydrostatic_strain_shifts_peaks(self, eps):
        """Hydrostatic strain: peaks shift from a=1 to a=1+eps."""
        _, perfect_pts = freud.data.UnitCell.sc().generate_system(4)
        N = len(perfect_pts)
        strained = self._apply_strain(perfect_pts, [eps, eps, eps])

        k_old = crystal_kvecs(1.0, K=3)
        k_new = crystal_kvecs(1 + eps, K=3)

        sk_at_old = sf3d(strained.astype(np.float32), k_old)
        sk_at_new = sf3d(strained.astype(np.float32), k_new)

        nonzero_new = np.any(k_new != 0, axis=1)
        bragg_old = self._bragg_indices(k_old)

        np.testing.assert_allclose(
            sk_at_new[nonzero_new],
            N,
            rtol=RTOL,
            err_msg=f"eps={eps}: strained crystal peaks at new positions should equal N",
        )
        np.testing.assert_array_less(
            np.mean(sk_at_old[bragg_old]),
            np.mean(sk_at_new[nonzero_new]),
            err_msg="Old peak positions should be weaker than new after strain",
        )

    @pytest.mark.parametrize("axis_idx", [0, 1, 2])
    def test_uniaxial_strain_shifts_peaks(self, axis_idx):
        """Uniaxial strain along each axis shifts peaks in that direction only."""
        _, perfect_pts = freud.data.UnitCell.sc().generate_system(4)
        N = len(perfect_pts)
        eps_diag = [0.0, 0.0, 0.0]
        eps_diag[axis_idx] = 0.05
        strained = self._apply_strain(perfect_pts, eps_diag)

        a_vals = [1.05 if i == axis_idx else 1.0 for i in range(3)]
        k_new = orthorhombic_kvecs(*a_vals, K=3)
        k_old = crystal_kvecs(1.0, K=3)

        sk_new = sf3d(strained.astype(np.float32), k_new)
        sk_old = sf3d(strained.astype(np.float32), k_old)

        nonzero_new = np.any(k_new != 0, axis=1)
        bragg_old = self._bragg_indices(k_old)

        np.testing.assert_allclose(
            sk_new[nonzero_new],
            N,
            rtol=RTOL,
            err_msg=f"axis={axis_idx}: peaks at new orthorhombic positions should equal N",
        )
        np.testing.assert_array_less(
            np.mean(sk_old[bragg_old]),
            np.mean(sk_new[nonzero_new]),
            err_msg=f"axis={axis_idx}: old cubic peaks should be weaker after uniaxial strain",
        )

    @pytest.mark.parametrize(
        "strain_diag",
        [
            [0.05, 0.05, 0.05],
            [0.05, 0.0, 0.0],
            [0.0, 0.05, 0.0],
            [0.0, 0.0, -0.03],
            [0.05, -0.02, 0.03],
        ],
    )
    def test_strain_preserves_s0(self, strain_diag):
        """S(0) = N is invariant under affine transformation."""
        _, perfect_pts = freud.data.UnitCell.sc().generate_system(4)
        N = len(perfect_pts)
        strained = self._apply_strain(perfect_pts, strain_diag)

        k_zero = np.array([[0, 0, 0]], dtype=np.float32)
        sk = sf3d(strained.astype(np.float32), k_zero)
        np.testing.assert_allclose(
            sk[0],
            N,
            rtol=1e-3,
            err_msg=f"strain={strain_diag}: S(0) = {sk[0]:.2f}, expected N={N}",
        )
