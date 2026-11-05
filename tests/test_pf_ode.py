"""The probability-flow ODE integrator of the OU variant.

With u = 0 the PF-ODE reduces to the OU mean flow dx = -1/2 sigma(t)^2 x dt,
whose exact solution is x(1) = x(0) exp(-1/2 int_0^1 sigma^2) with
int_0^1 sigma^2 = sigma_T_0^2 (scheduler identity, tested separately).
The explicit-Euler chain converges to this at O(dt); the test asserts the
value at a resolved step count and the first-order error decay.
"""

import jax
import jax.numpy as jnp
import numpy as np

from src.networks.amdp.jax_integrators_ou import ode_integrator
from src.networks.amdp.jax_policy_ou import OUSDE
from src.networks.amdp.jax_scheduler import get_geometric_scheduler
from tests.conftest import ACTION_DIM, OBS_DIM

from flax import nnx

# A gentle schedule keeps 1/2 sigma^2 dt << 1 so explicit Euler is resolved.
SIGMA_MAX, SIGMA_MIN = 1.0, 0.5


def _integrate(diff_steps, x0):
    sched = get_geometric_scheduler(SIGMA_MAX, SIGMA_MIN)
    model = OUSDE(
        action_dim=ACTION_DIM,
        observation_dim=OBS_DIM,
        fwd_model=None,
        diff_steps=diff_steps,
        scheduler=sched,
        rngs=nnx.Rngs(0),
    )
    integrate = ode_integrator(jnp.zeros(OBS_DIM), model)
    state = (x0, jax.random.PRNGKey(0))
    state, _ = jax.lax.scan(integrate, state, jnp.arange(diff_steps))
    return np.asarray(state[0], dtype=np.float64), sched


def test_zero_control_flow_is_ou_mean_contraction():
    x0 = jnp.array([1.0, -2.0, 0.5])
    xT_coarse, sched = _integrate(64, x0)
    xT_fine, _ = _integrate(512, x0)

    factor = np.exp(-0.5 * float(sched.sigma_T_0()) ** 2)
    expected = np.asarray(x0, dtype=np.float64) * factor

    err_coarse = np.max(np.abs(xT_coarse - expected) / np.abs(expected))
    err_fine = np.max(np.abs(xT_fine - expected) / np.abs(expected))

    assert err_fine < 1.5e-3, f"512-step flow off by {err_fine:.2e}"
    assert err_fine < err_coarse / 4.0, (
        f"no first-order decay: {err_coarse:.2e} -> {err_fine:.2e}"
    )


def test_ode_sample_is_deterministic_in_the_key(make_ou_actor):
    """Same key => identical actions; the only randomness is the prior draw."""
    actor = make_ou_actor(fwd_model=None)
    obs = jax.random.normal(jax.random.PRNGKey(21), (8, OBS_DIM))
    a1 = actor.ode_sample(jax.random.PRNGKey(5), obs, stop_grad=True)
    a2 = actor.ode_sample(jax.random.PRNGKey(5), obs, stop_grad=True)
    a3 = actor.ode_sample(jax.random.PRNGKey(6), obs, stop_grad=True)
    np.testing.assert_array_equal(np.asarray(a1), np.asarray(a2))
    assert not np.array_equal(np.asarray(a1), np.asarray(a3))
