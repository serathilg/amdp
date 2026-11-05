"""API contract of DiffusionActor.sde_sample.

Shape conventions (B = batch, d = action dim): the four sample tensors are
(B, d); the log-quantities keep the integrator's per-sample (1,) axis, so
(B, 1); squashing_correction_val is the value_and_grad scalar, so (B,).
Randomness contract: the key is split once per batch row inside sde_sample,
so rows are independent -- editing one row's observation must leave every
other row's outputs bit-identical (this is what a loop-of-singles comparison
cannot see, because the internal split binds keys to row positions).
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx

from tests.conftest import ACTION_DIM, OBS_DIM, ToyControl


@pytest.fixture(params=["brownian", "ou"])
def controlled_actor(request, make_brownian_actor, make_ou_actor):
    make = make_brownian_actor if request.param == "brownian" else make_ou_actor
    return make(fwd_model=ToyControl())


def test_field_shapes(controlled_actor):
    B = 6
    obs = jax.random.normal(jax.random.PRNGKey(0), (B, OBS_DIM))
    s = controlled_actor.sde_sample(jax.random.PRNGKey(1), obs, stop_grad=True)

    assert s.action.shape == (B, ACTION_DIM)
    assert s.action_unsquashed.shape == (B, ACTION_DIM)
    assert s.prior_action.shape == (B, ACTION_DIM)
    assert s.squashing_correction_grad.shape == (B, ACTION_DIM)
    assert s.log_weight.shape == (B, 1)
    assert s.log_path_weight_deterministic.shape == (B, 1)
    assert s.log_path_weight_stochastic.shape == (B, 1)
    assert s.log_p_T_ref.shape == (B, 1)
    assert s.squashing_correction_val.shape == (B,)
    assert all(jnp.all(jnp.isfinite(f)) for f in s)


def test_deterministic_in_key(controlled_actor):
    obs = jax.random.normal(jax.random.PRNGKey(2), (5, OBS_DIM))
    s1 = controlled_actor.sde_sample(jax.random.PRNGKey(3), obs, stop_grad=True)
    s2 = controlled_actor.sde_sample(jax.random.PRNGKey(3), obs, stop_grad=True)
    for name, f1, f2 in zip(s1._fields, s1, s2, strict=True):
        np.testing.assert_array_equal(
            np.asarray(f1), np.asarray(f2), err_msg=f"field {name} not deterministic"
        )


def test_rows_are_independent(controlled_actor):
    """Perturbing row j leaves all rows i != j bit-identical."""
    obs = jax.random.normal(jax.random.PRNGKey(4), (5, OBS_DIM))
    obs_mod = obs.at[2].add(1.0)
    key = jax.random.PRNGKey(5)

    s = controlled_actor.sde_sample(key, obs, stop_grad=True)
    s_mod = controlled_actor.sde_sample(key, obs_mod, stop_grad=True)

    keep = np.array([0, 1, 3, 4])
    for name, f, f_mod in zip(s._fields, s, s_mod, strict=True):
        np.testing.assert_array_equal(
            np.asarray(f)[keep],
            np.asarray(f_mod)[keep],
            err_msg=f"row leakage in field {name}",
        )
    # The perturbed row does change (the control is observation-dependent).
    assert not np.array_equal(
        np.asarray(s.action)[2], np.asarray(s_mod.action)[2]
    )


def test_jit_matches_eager(controlled_actor):
    obs = jax.random.normal(jax.random.PRNGKey(6), (4, OBS_DIM))
    key = jax.random.PRNGKey(7)

    eager = controlled_actor.sde_sample(key, obs, stop_grad=True)

    @nnx.jit
    def jitted(actor, key_, obs_):
        return actor.sde_sample(key_, obs_, stop_grad=True)

    jit_s = jitted(controlled_actor, key, obs)
    for name, f_e, f_j in zip(eager._fields, eager, jit_s, strict=True):
        np.testing.assert_allclose(
            np.asarray(f_e),
            np.asarray(f_j),
            rtol=1e-6,
            atol=1e-6,
            err_msg=f"jit/eager mismatch in field {name}",
        )
