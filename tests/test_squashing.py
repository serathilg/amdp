"""Squashing bijectors and the correction terms reported by sde_sample.

Erf: f(x) = erf(k x), log|f'| = log(2k) - 1/2 log(pi) - (k x)^2, and
d/dx log|f'| = -2 k^2 x. Tanh: d/dx log|tanh'| = -2 tanh(x). The analytic
gradients cross-check sde_sample's autodiff-computed
squashing_correction_grad without re-running the same autodiff.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from src.networks.amdp.jax_policy import Erf as BrownianErf
from src.networks.amdp.jax_policy_ou import Erf as OUErf
from tests.conftest import OBS_DIM


def _erf_k(actor):
    return float(actor.squashing._k)


@pytest.mark.parametrize("erf_cls_key", ["brownian", "ou"])
def test_erf_log_det_matches_autodiff(erf_cls_key, scheduler):
    bij = BrownianErf(scheduler) if erf_cls_key == "brownian" else OUErf(scheduler)
    x = jnp.linspace(-2.0, 2.0, 41)

    fwd_grad = jax.vmap(jax.grad(bij.forward))(x)
    np.testing.assert_allclose(
        np.asarray(bij.forward_log_det_jacobian(x)),
        np.log(np.abs(np.asarray(fwd_grad))),
        rtol=1e-5,
        atol=1e-6,
    )


@pytest.mark.parametrize("erf_cls_key", ["brownian", "ou"])
def test_erf_inverse_roundtrip(erf_cls_key, scheduler):
    bij = BrownianErf(scheduler) if erf_cls_key == "brownian" else OUErf(scheduler)
    # Stay clear of erf saturation so float32 erfinv is well-conditioned.
    k = float(bij._k)
    x = jnp.linspace(-1.5 / k, 1.5 / k, 31)
    np.testing.assert_allclose(
        np.asarray(bij.inverse(bij.forward(x))), np.asarray(x), rtol=2e-3, atol=2e-4
    )


def test_ou_erf_uses_unit_variance_k(scheduler, make_ou_actor):
    """The OU Erf assumes a unit-variance terminal reference: k = sqrt(1/2)."""
    actor = make_ou_actor()
    np.testing.assert_allclose(_erf_k(actor), np.sqrt(0.5), rtol=1e-6)


def test_brownian_erf_k_matches_reference_std(scheduler, make_brownian_actor):
    """The Brownian Erf matches its N(0, sigma_T_0^2) terminal reference."""
    actor = make_brownian_actor()
    np.testing.assert_allclose(
        _erf_k(actor),
        1.0 / (float(scheduler.sigma_T_0()) * np.sqrt(2.0)),
        rtol=1e-6,
    )


@pytest.mark.parametrize("squashing", ["erf", "tanh"])
def test_squashing_correction_grad_is_analytic(squashing, make_brownian_actor):
    """sde_sample's squashing_correction_grad equals the analytic gradient
    of sum_i log|f'(x_i)| at x = action_unsquashed."""
    actor = make_brownian_actor(squashing=squashing)
    obs = jax.random.normal(jax.random.PRNGKey(11), (32, OBS_DIM))
    sample = actor.sde_sample(jax.random.PRNGKey(12), obs, stop_grad=True)
    xu = np.asarray(sample.action_unsquashed)

    if squashing == "erf":
        k = _erf_k(actor)
        expected = -2.0 * k**2 * xu
    else:
        expected = -2.0 * np.tanh(xu)

    np.testing.assert_allclose(
        np.asarray(sample.squashing_correction_grad), expected, rtol=1e-4, atol=1e-5
    )
    # And the reported value is the summed log-det at the same point.
    expected_val = np.asarray(
        actor.squashing.forward_log_det_jacobian(jnp.asarray(xu)).sum(axis=-1)
    )
    np.testing.assert_allclose(
        np.asarray(sample.squashing_correction_val), expected_val, rtol=1e-5, atol=1e-5
    )


def test_action_is_squashed_unsquashed(make_brownian_actor):
    """action == f(action_unsquashed) for both bijectors."""
    for squashing in ("erf", "tanh"):
        actor = make_brownian_actor(squashing=squashing)
        obs = jax.random.normal(jax.random.PRNGKey(13), (8, OBS_DIM))
        s = actor.sde_sample(jax.random.PRNGKey(14), obs, stop_grad=True)
        np.testing.assert_allclose(
            np.asarray(s.action),
            np.asarray(actor.squashing.forward(s.action_unsquashed)),
            rtol=1e-6,
            atol=1e-7,
        )
