"""Zero-control terminal marginals of the SDE integrators.

With u = 0 both Euler-Maruyama chains are linear-Gaussian, so X_T is exactly
Gaussian and its variance follows from the scheme itself (a property of the
discretization, not a value captured from the code):

Brownian (X_0 = 0):        X_T = sum_k sigma(t_k) sqrt(dt) eps_k
                           => Var = sum_k sigma(t_k)^2 dt         (left Riemann)
OU (X_0 ~ N(0,1)):         x_{k+1} = x_k (1 - 1/2 sigma_k^2 dt) + sigma_k sqrt(dt) eps_k
                           => v_{k+1} = v_k (1 - 1/2 sigma_k^2 dt)^2 + sigma_k^2 dt

The continuum limits are Var -> sigma_T_0^2 (Brownian) and v -> 1 (OU
stationarity); at coarse diff_steps the discrete values differ substantially
(for the default scheduler 1/2 sigma(0)^2 dt > 1 at 16 steps), so the sampled
statistics are compared against the discrete recursions tightly, and the
continuum limit is asserted only as a convergence property.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from src.networks.amdp.jax_scheduler import get_geometric_scheduler
from tests.conftest import ACTION_DIM, OBS_DIM

N_SAMPLES = 8192


def _sigma_sq_grid(scheduler, diff_steps):
    """sigma(t_k)^2 at the left endpoints t_k = k/K used by the integrators."""
    t = np.arange(diff_steps) / diff_steps
    return np.asarray(scheduler.sigma_t(jnp.asarray(t)), dtype=np.float64) ** 2


def _sample_unsquashed(actor, n):
    obs = jnp.zeros((n, OBS_DIM))
    sample = actor.sde_sample(jax.random.PRNGKey(7), obs, stop_grad=True)
    return np.asarray(sample.action_unsquashed, dtype=np.float64)


def _check_gaussian(x, expected_var):
    """Mean, variance, and a 1-sigma-mass normality check, 5-sigma bands."""
    n = x.size
    np.testing.assert_allclose(x.mean(), 0.0, atol=5 * np.sqrt(expected_var / n))
    np.testing.assert_allclose(
        x.var(), expected_var, rtol=5 * np.sqrt(2.0 / n)
    )
    frac_1sigma = np.mean(np.abs(x) < np.sqrt(expected_var))
    p = 0.682689
    np.testing.assert_allclose(
        frac_1sigma, p, atol=5 * np.sqrt(p * (1 - p) / n)
    )


def test_brownian_terminal_marginal(scheduler, make_brownian_actor):
    actor = make_brownian_actor()
    diff_steps = actor.diffusion_model.diff_steps
    expected_var = float(_sigma_sq_grid(scheduler, diff_steps).sum() / diff_steps)
    x = _sample_unsquashed(actor, N_SAMPLES)
    _check_gaussian(x, expected_var)


def test_ou_terminal_marginal(scheduler, make_ou_actor):
    actor = make_ou_actor()
    diff_steps = actor.diffusion_model.diff_steps
    dt = 1.0 / diff_steps
    v = 1.0
    for s_sq in _sigma_sq_grid(scheduler, diff_steps):
        v = v * (1.0 - 0.5 * s_sq * dt) ** 2 + s_sq * dt
    x = _sample_unsquashed(actor, N_SAMPLES)
    _check_gaussian(x, v)


def test_brownian_variance_converges_to_sigma_T_0(scheduler):
    """Left-Riemann variance -> sigma_T_0^2 at O(dt)."""
    target = float(scheduler.sigma_T_0()) ** 2
    gaps = []
    for diff_steps in (16, 128, 1024):
        var = _sigma_sq_grid(scheduler, diff_steps).sum() / diff_steps
        gaps.append(abs(var - target) / target)
    assert gaps[2] < gaps[1] < gaps[0]
    # O(dt) rate: the leading error term dt * (sigma(0)^2 - sigma(1)^2) / 2
    # predicts ~5.2e-3 at 1024 steps and a 64x decay from 16 steps.
    assert gaps[2] < 6e-3
    assert gaps[0] / gaps[2] > 30


@pytest.mark.slow
def test_ou_stationarity_in_the_continuum_limit(make_ou_actor):
    """With a resolved discretization the OU chain preserves N(0, 1)."""
    sched = get_geometric_scheduler(1.0, 0.5)
    actor = make_ou_actor(diff_steps=512, sched=sched)
    dt = 1.0 / 512
    v = 1.0
    for s_sq in _sigma_sq_grid(sched, 512):
        v = v * (1.0 - 0.5 * s_sq * dt) ** 2 + s_sq * dt
    assert abs(v - 1.0) < 0.01, "discrete variance should stay near stationary"
    x = _sample_unsquashed(actor, N_SAMPLES)
    _check_gaussian(x, v)
    np.testing.assert_allclose(x.var(), 1.0, rtol=0.03)


def test_prior_action_matches_prior(make_brownian_actor, make_ou_actor):
    """prior_action is the Dirac zero (Brownian) / a N(0,I) draw (OU)."""
    obs = jnp.zeros((256, OBS_DIM))
    b = make_brownian_actor().sde_sample(jax.random.PRNGKey(0), obs, stop_grad=True)
    np.testing.assert_array_equal(np.asarray(b.prior_action), 0.0)

    o = make_ou_actor().sde_sample(jax.random.PRNGKey(0), obs, stop_grad=True)
    prior = np.asarray(o.prior_action, dtype=np.float64)
    n = prior.size
    np.testing.assert_allclose(prior.mean(), 0.0, atol=5 / np.sqrt(n))
    np.testing.assert_allclose(prior.var(), 1.0, rtol=5 * np.sqrt(2.0 / n))
    assert prior.shape == (256, ACTION_DIM)
