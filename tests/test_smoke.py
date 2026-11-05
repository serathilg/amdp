import jax
import jax.numpy as jnp

from src.networks.amdp.types import SDESample


def test_sde_sample_field_order():
    """SDESample's fields are the documented 9-tuple, in order.

    The trainers pack these into Transition by name; the tuple order is part
    of the public API.
    """
    assert SDESample._fields == (
        "action",
        "action_unsquashed",
        "prior_action",
        "squashing_correction_grad",
        "log_weight",
        "log_path_weight_deterministic",
        "log_path_weight_stochastic",
        "log_p_T_ref",
        "squashing_correction_val",
    )


def test_zero_control_sample_runs(make_brownian_actor):
    """A zero-control actor draws finite samples inside (-1, 1)."""
    actor = make_brownian_actor()
    obs = jnp.zeros((4, 5))
    sample = actor.sde_sample(jax.random.PRNGKey(0), obs, stop_grad=True)
    assert sample.action.shape == (4, 3)
    assert jnp.all(jnp.isfinite(sample.action))
    assert jnp.all(jnp.abs(sample.action) < 1.0)
