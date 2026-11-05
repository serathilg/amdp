import math
from functools import partial
from typing import NamedTuple

import distrax
import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from jaxtyping import Array, Float, PRNGKeyArray

from src.networks.amdp.types import SDESample


class OUSDE(nnx.Module):
    """Controlled SDE with an Ornstein-Uhlenbeck reference and N(0, I) prior."""

    def __init__(
        self,
        action_dim: int,
        observation_dim: int,
        fwd_model: nnx.Module = None,
        diff_steps: int = 8,
        scheduler: NamedTuple = None,
        *,
        rngs: nnx.Rngs,
    ):
        self.action_dim = action_dim
        self.observation_dim = observation_dim
        self.diff_steps = diff_steps
        self.fwd_model = fwd_model
        self.noise_scheduler = scheduler

        dt = 1. / diff_steps
        self.dt = dt

        # The N(0, I) reference prior is built inside the closures rather than captured
        # from here: constructing it stages array constants into whichever trace runs
        # __init__, and those would escape that trace inside this module's graphdef.
        def _prior():
            return distrax.MultivariateNormalDiag(
                np.zeros(action_dim), np.ones(action_dim)
            )

        @jax.jit
        def _ref_log_prob_fn(x):
            log_probs = _prior().log_prob(x)
            return log_probs

        @partial(jax.jit, static_argnames=["n_samples"])
        def _prior_sampler_fn(key, n_samples):
            samples = _prior().sample(seed=key, sample_shape=(n_samples,))
            return samples

        self.ref_log_prob = _ref_log_prob_fn
        self.prior_sampler = _prior_sampler_fn

    def forward_model(
        self,
        step: Float[Array, ""],
        x: Float[Array, "act"],
        obs: Float[Array, "obs"],
        aux: jax.Array = None,
    ) -> Float[Array, "act"]:
        """Per-sample control u(x, obs, t); called inside the integrator scan."""
        if self.fwd_model is not None:
            return self.fwd_model(x, obs, step)
        else:
            return jnp.zeros_like(x)


class Erf(distrax.Bijector):
    """Bijector that computes y = erf(k * x), mapping R to (-1, 1).
    
    Always uses k = sqrt(1/2) as unit-variance p_T^ref is assumed."""
    
    def __init__(self, noise_scheduler):
        # event_ndims_in=0 indicates this is an element-wise transformation
        super().__init__(event_ndims_in=0)
        self._noise_scheduler = noise_scheduler

    @property
    def _k(self):
        return jnp.sqrt(0.5)
    
    def forward(self, x: Float[Array, "*shape"]) -> Float[Array, "*shape"]:
        return jax.scipy.special.erf(self._k * x)

    def forward_log_det_jacobian(
        self, x: Float[Array, "*shape"]
    ) -> Float[Array, "*shape"]:
        # f'(x) = (2k / sqrt(pi)) * exp(-(kx)^2)
        # log|f'(x)| = log(2k) - 0.5*log(pi) - (kx)^2
        # nabla log|f'(x)| = - 2 k^2 x
        return jnp.log(2 * self._k) - 0.5 * jnp.log(jnp.pi) - (self._k * x)**2

    def forward_and_log_det(self, x):
        # Implementing both at once is often more efficient for Distrax
        return self.forward(x), self.forward_log_det_jacobian(x)

    def inverse(self, y: Float[Array, "*shape"]) -> Float[Array, "*shape"]:
        # Maps (-1, 1) back to R
        return jax.scipy.special.erfinv(y) / self._k

class DiffusionActor(nnx.Module):
    def __init__(
        self,
        action_dim: int,
        observation_dim: int,
        diffusion_model: nnx.Module,
        sde_integrator: callable,
        ode_integrator: callable,
        kl_start: float = 0.1,
        ent_start: float = 0.1,
        kl_bound:float = 0.1,
        squashing: str = "erf",
    ):
        self.action_dim = action_dim
        self.observation_dim = observation_dim
        self.diffusion_model = diffusion_model
        self.sde_integrator = sde_integrator
        self.ode_integrator = ode_integrator

        # Parameters
        self.log_lagrangian = nnx.Param(jnp.ones(1) * math.log(kl_start))
        self.log_temperature = nnx.Param(jnp.ones(1) * math.log(ent_start))

        if squashing == "erf":
            self.squashing = Erf(self.diffusion_model.noise_scheduler)
        elif squashing == "tanh":
            self.squashing = distrax.Tanh()
        else:
            raise ValueError(f"Unknown squashing bijector: {squashing!r} (expected 'erf' or 'tanh')")

    def ode_sample(
        self,
        key: PRNGKeyArray,
        obs: Float[Array, "batch obs"],
        stop_grad: bool = False,
    ) -> Float[Array, "batch act"]:
        """Sample actions deterministically via the probability-flow ODE."""
        keys = jax.random.split(key, num=obs.shape[0])
        
        def _ode_sample(key, obs):
            key, key_init, key_aux = jax.random.split(key, 3)
            init_x = self.diffusion_model.prior_sampler(key_init, 1)
            init_x = jnp.squeeze(init_x, 0)
            if stop_grad:
                init_x = jax.lax.stop_gradient(init_x)
            aux = (init_x, key_aux)
            
            integrate = self.ode_integrator(obs, self.diffusion_model, stop_grad)
            aux, _ = jax.lax.scan(integrate, aux, jnp.arange(0, self.diffusion_model.diff_steps))
            final_x, _ = aux
            
            final_x = self.squashing.forward(final_x)
            return final_x
        
        x_0 = jax.vmap(_ode_sample, in_axes=(0, 0))(keys, obs)
        return x_0

    def sde_sample(
        self,
        key: PRNGKeyArray,
        obs: Float[Array, "batch obs"],
        stop_grad: bool = False,
    ) -> SDESample:
        """Sample actions from the SDE diffusion model."""
        keys = jax.random.split(key, num=obs.shape[0])

        def squashing_correction(x):
            return self.squashing.forward_log_det_jacobian(x).sum()

        def _sde_sample(key, obs):
            key, key_init, key_aux = jax.random.split(key, 3)
            init_x = self.diffusion_model.prior_sampler(key_init, 1)
            init_x = jnp.squeeze(init_x, 0)
            if stop_grad:
                init_x = jax.lax.stop_gradient(init_x)
            aux = (init_x, jnp.zeros(1), jnp.zeros(1), key_aux)
            
            integrate = self.sde_integrator(obs, self.diffusion_model, stop_grad=stop_grad)
            aux, _ = jax.lax.scan(integrate, aux, jnp.arange(0, self.diffusion_model.diff_steps))
            final_x_unsquashed, log_path_weight_deterministic, log_path_weight_stochastic, _ = aux
            final_x = self.squashing.forward(final_x_unsquashed)
            log_p_T_ref = self.diffusion_model.ref_log_prob(final_x_unsquashed).reshape(log_path_weight_deterministic.shape)

            squashing_correction_val, squashing_correction_grad = jax.value_and_grad(squashing_correction)(final_x_unsquashed)
            # Single-path estimate of -log pi(a|s) whose expectation lower-bounds the
            # policy entropy (paper Prop. 4.1 / App. E.4):
            #   log dPi^0/dPi^u - log Pi^0_1(X_1) + log|det J_f(X_1)|
            # with a = f(X_1), where X_1 is the terminal sample (code step T = tau=1)
            # and log_p_T_ref stores log Pi^0_1. log dPi^0/dPi^u are the accumulated
            # Girsanov terms -int 1/2 ||u||^2 dt - int u.dB (the stochastic integral
            # has zero mean). The controlled and the reference OU process share the
            # N(0, I) initial law, so the boundary densities cancel in the RND -- no
            # p_prior(X_0) term, exactly as in the Brownian-reference variant.
            log_weight = log_path_weight_deterministic + log_path_weight_stochastic - log_p_T_ref + squashing_correction_val
            return SDESample(
                action=final_x,
                action_unsquashed=final_x_unsquashed,
                prior_action=init_x,
                squashing_correction_grad=squashing_correction_grad,
                log_weight=log_weight,
                log_path_weight_deterministic=log_path_weight_deterministic,
                log_path_weight_stochastic=log_path_weight_stochastic,
                log_p_T_ref=log_p_T_ref,
                squashing_correction_val=squashing_correction_val,
            )

        return jax.vmap(_sde_sample, in_axes=(0, 0))(keys, obs)


    def temperature(self) -> jax.Array:
        return jnp.exp(self.log_temperature.value)

    def lagrangian(self) -> jax.Array:
        return jnp.exp(self.log_lagrangian.value)