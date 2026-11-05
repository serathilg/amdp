"""Identities of the geometric noise scheduler.

The scheduler defines sigma_t(t) = sigma_min * r^(1-t) * sqrt(2 log r) with
r = sigma_max/sigma_min, and claims sigma_t_0(t)^2 = int_0^t sigma_t(v)^2 dv
(closed form sigma_max^2 (1 - r^(-2t))). The bridge quantities follow from
Gaussian conditioning of Brownian motion on (X_0, X_T).
"""

import jax.numpy as jnp
import numpy as np

from tests.conftest import SIGMA_MAX, SIGMA_MIN


def test_sigma_t_0_is_cumulative_variance(scheduler):
    """sigma_t_0(t)^2 equals the integral of sigma_t(v)^2 from 0 to t.

    Trapezoid quadrature on a dense grid in float64; the integrand is smooth,
    so the quadrature error is far below the float32 rounding of the scheduler
    outputs (rtol 5e-4).
    """
    v = np.linspace(0.0, 1.0, 20001)
    integrand = np.asarray(scheduler.sigma_t(jnp.asarray(v)), dtype=np.float64) ** 2
    for t_idx in [2000, 5000, 10000, 15000, 20000]:
        integral = np.trapezoid(integrand[: t_idx + 1], v[: t_idx + 1])
        claimed = float(scheduler.sigma_t_0(v[t_idx])) ** 2
        np.testing.assert_allclose(claimed, integral, rtol=5e-4)


def test_sigma_T_0_is_terminal_sigma_t_0(scheduler):
    assert float(scheduler.sigma_T_0()) == float(scheduler.sigma_t_0(1.0))


def test_bridge_identities(scheduler):
    """Brownian-bridge conditionals derived from sigma_t_0.

    sigma_t_0T(t)^2 = sigma_t_0(t)^2 (1 - sigma_t_0(t)^2/sigma_T_0^2)
    noise_0T_scale = sigma_t_0T (both equal sigma_t_0 * sigma_t_T / sigma_T_0)
    sigma_t_T(t)^2 = sigma_T_0^2 - sigma_t_0(t)^2
    """
    t = jnp.linspace(0.05, 0.95, 19)
    s_t0 = scheduler.sigma_t_0(t)
    s_T0 = scheduler.sigma_T_0()

    # atol at float32-ulp scale: (1 - ratio^2) cancels catastrophically as
    # sigma_t_0 -> sigma_T_0, so the relative error is unbounded at late t
    # while the absolute error stays at machine precision.
    np.testing.assert_allclose(
        scheduler.sigma_t_0T(t) ** 2,
        s_t0**2 * (1.0 - s_t0**2 / s_T0**2),
        rtol=1e-5,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        scheduler.noise_0T_scale(t), scheduler.sigma_t_0T(t), rtol=1e-4, atol=1e-5
    )
    np.testing.assert_allclose(
        scheduler.sigma_t_T(t) ** 2, s_T0**2 - s_t0**2, rtol=1e-4, atol=1e-6
    )


def test_bridge_mean_scales_sum_to_one(scheduler):
    """mu_t_0T_scale == xT_0T_scale and x0_0T_scale = 1 - xT_0T_scale."""
    t = jnp.linspace(0.0, 1.0, 21)
    np.testing.assert_array_equal(
        scheduler.mu_t_0T_scale(t), scheduler.xT_0T_scale(t)
    )
    np.testing.assert_allclose(
        scheduler.xT_0T_scale(t) + scheduler.x0_0T_scale(t), 1.0, rtol=1e-6
    )


def test_parameters_respected():
    """A scheduler built directly from SIGMA_MAX/SIGMA_MIN honors the closed form
    sigma_T_0 = sqrt(sigma_max^2 (1 - r^-2)) (-> sigma_max as r -> inf)."""
    from src.networks.amdp.jax_scheduler import get_geometric_scheduler

    sched = get_geometric_scheduler(SIGMA_MAX, SIGMA_MIN)
    r = SIGMA_MAX / SIGMA_MIN
    np.testing.assert_allclose(
        float(sched.sigma_T_0()),
        np.sqrt(SIGMA_MAX**2 * (1 - r ** (-2.0))),
        rtol=1e-6,
    )
