"""Zero-control log-weight identities of sde_sample.

With u = 0 the Girsanov terms vanish identically, so
    log_weight = -log Pi^0_1(X_1) + sum_i log|f'(X_1_i)|.
For the scaled-erf squashing f(x) = erf(k x) with k = 1/(sigma_ref sqrt(2))
matched to the reference terminal marginal N(0, sigma_ref^2 I), the
pushforward of the reference through f is Uniform(-1,1)^d, hence pointwise
(for every X_1, not just in expectation):
    log_weight = -log(2^-d) = d log 2.
Brownian variant: sigma_ref = sigma_T_0 (Erf reads it off the scheduler).
OU variant: sigma_ref = 1 with k = sqrt(1/2).
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx

from tests.conftest import ACTION_DIM, OBS_DIM


@pytest.mark.parametrize("variant", ["brownian", "ou"])
def test_zero_control_log_weight_identity(variant, make_brownian_actor, make_ou_actor):
    actor = make_brownian_actor() if variant == "brownian" else make_ou_actor()
    obs = jax.random.normal(jax.random.PRNGKey(1), (64, OBS_DIM))
    sample = actor.sde_sample(jax.random.PRNGKey(2), obs, stop_grad=True)

    # Girsanov terms are identically zero, not merely small.
    np.testing.assert_array_equal(
        np.asarray(sample.log_path_weight_deterministic), 0.0
    )
    np.testing.assert_array_equal(np.asarray(sample.log_path_weight_stochastic), 0.0)

    # Pointwise uniform-pushforward identity. The |X_T|^2 terms of the
    # reference log-density and the squashing correction cancel analytically;
    # in float32 the cancellation leaves ~1e-5 residue.
    np.testing.assert_allclose(
        np.asarray(sample.log_weight),
        ACTION_DIM * np.log(2.0),
        rtol=1e-4,
    )


@pytest.mark.parametrize("variant", ["brownian", "ou"])
def test_zero_control_log_weight_identity_under_jit(
    variant, make_brownian_actor, make_ou_actor
):
    """The identity and jit-vs-eager parity of the full SDESample."""
    actor = make_brownian_actor() if variant == "brownian" else make_ou_actor()
    obs = jax.random.normal(jax.random.PRNGKey(3), (16, OBS_DIM))
    key = jax.random.PRNGKey(4)

    eager = actor.sde_sample(key, obs, stop_grad=True)

    @nnx.jit(static_argnames=())
    def jitted(actor_, key_, obs_):
        return actor_.sde_sample(key_, obs_, stop_grad=True)

    jit_sample = jitted(actor, key, obs)

    np.testing.assert_allclose(
        np.asarray(jit_sample.log_weight), ACTION_DIM * np.log(2.0), rtol=1e-4
    )
    for name, eager_field, jit_field in zip(
        eager._fields, eager, jit_sample, strict=True
    ):
        np.testing.assert_allclose(
            np.asarray(eager_field),
            np.asarray(jit_field),
            rtol=1e-6,
            atol=1e-6,
            err_msg=f"jit/eager mismatch in field {name}",
        )
