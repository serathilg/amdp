import os

# Force CPU before any jax import: the unit tests must not grab a GPU and must
# behave the same on machines without one.
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax.numpy as jnp
import pytest
from flax import nnx

from src.networks.amdp.jax_integrators import (
    sde_integrator as brownian_sde_integrator,
)
from src.networks.amdp.jax_integrators_ou import (
    ode_integrator as ou_ode_integrator,
)
from src.networks.amdp.jax_integrators_ou import (
    sde_integrator as ou_sde_integrator,
)
from src.networks.amdp.jax_policy import BrownianSDE
from src.networks.amdp.jax_policy import DiffusionActor as BrownianActor
from src.networks.amdp.jax_policy_ou import OUSDE
from src.networks.amdp.jax_policy_ou import DiffusionActor as OUActor
from src.networks.amdp.jax_scheduler import get_geometric_scheduler

# Small dimensions keep every test fast; OBS_DIM >= ACTION_DIM so ToyControl
# can slice the observation.
ACTION_DIM = 3
OBS_DIM = 5
DIFF_STEPS = 16
SIGMA_MAX = 2.0
SIGMA_MIN = 0.01


class ToyControl(nnx.Module):
    """Deterministic nonlinear control with the ControlNetwork call signature.

    Parameter-free on purpose: tests need a nontrivial, reproducible control,
    not a trainable one.
    """

    def __call__(self, actions, observations, time):
        return (
            jnp.tanh(actions)
            + 0.1 * observations[..., : actions.shape[-1]]
            + 0.01 * time
        )


@pytest.fixture(scope="session")
def scheduler():
    return get_geometric_scheduler(SIGMA_MAX, SIGMA_MIN)


@pytest.fixture
def make_brownian_actor(scheduler):
    """Factory for a Brownian-reference DiffusionActor.

    fwd_model=None gives exact zero control (BrownianSDE.forward_model then
    returns jnp.zeros_like(x)), which the zero-control identities rely on.
    """

    def _make(fwd_model=None, squashing="erf", diff_steps=DIFF_STEPS, sched=None):
        model = BrownianSDE(
            action_dim=ACTION_DIM,
            observation_dim=OBS_DIM,
            fwd_model=fwd_model,
            diff_steps=diff_steps,
            scheduler=sched if sched is not None else scheduler,
            rngs=nnx.Rngs(0),
        )
        return BrownianActor(
            action_dim=ACTION_DIM,
            observation_dim=OBS_DIM,
            diffusion_model=model,
            sde_integrator=brownian_sde_integrator,
            squashing=squashing,
        )

    return _make


@pytest.fixture
def make_ou_actor(scheduler):
    """Factory for an OU-reference DiffusionActor (zero control by default)."""

    def _make(fwd_model=None, squashing="erf", diff_steps=DIFF_STEPS, sched=None):
        model = OUSDE(
            action_dim=ACTION_DIM,
            observation_dim=OBS_DIM,
            fwd_model=fwd_model,
            diff_steps=diff_steps,
            scheduler=sched if sched is not None else scheduler,
            rngs=nnx.Rngs(0),
        )
        return OUActor(
            action_dim=ACTION_DIM,
            observation_dim=OBS_DIM,
            diffusion_model=model,
            sde_integrator=ou_sde_integrator,
            ode_integrator=ou_ode_integrator,
            squashing=squashing,
        )

    return _make
