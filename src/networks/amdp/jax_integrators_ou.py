import jax
import jax.numpy as jnp
from jaxtyping import Array, Float


# OU-reference integrators following Denoising Diffusion Samplers (DDS),
# Vargas et al., 2023, https://arxiv.org/abs/2302.13834
#
# Controlled SDE:   dX_t = -1/2 sigma(t)^2 X_t dt + sigma(t) u(X_t, s, t) dt + sigma(t) dB_t
# Uncontrolled OU:  dX_t = -1/2 sigma(t)^2 X_t dt + sigma(t) dB_t
# Both share the OU drift -1/2 sigma(t)^2 X_t, so Girsanov on the controlled measure relative
# to the uncontrolled one only uses u in the log path weight. ctrl below is the raw control u
# emitted by the forward model.


def sde_integrator(obs: Float[Array, "obs"], diffusion_model, stop_grad: bool = False):
    """Euler-Maruyama step closure for one sample path.

    Scan state: (x: Float[Array, "act"], log_path_weight_deterministic and
    log_path_weight_stochastic: Float[Array, "1"], key).
    """
    dt = diffusion_model.dt

    def integrate_EM(state, step):
        step = step.astype(jnp.float32)
        x, log_path_weight_deterministic, log_path_weight_stochastic, key_gen = state

        key, key_gen = jax.random.split(key_gen)
        noise = jax.random.normal(key, x.shape)
        t = step * dt

        sigma_t = diffusion_model.noise_scheduler.sigma_t(t)

        # Raw AMDP control u from the forward model
        ctrl = diffusion_model.forward_model(t, x, obs)

        # OU drift + control drift, diffusion sigma_t dB
        ou_drift = -0.5 * sigma_t ** 2 * x
        ctrl_drift = sigma_t * ctrl
        x_new = x + (ou_drift + ctrl_drift) * dt + sigma_t * noise * jnp.sqrt(dt)

        # Girsanov path weight log(dPi^0/dPi^u) with theta = u (since drift difference is sigma*u
        # and diffusion is sigma)
        log_path_weight_deterministic += -0.5 * jnp.sum(ctrl ** 2) * dt
        log_path_weight_stochastic += -jnp.sum(ctrl * noise) * jnp.sqrt(dt)

        next_state = (x_new, log_path_weight_deterministic, log_path_weight_stochastic, key_gen)
        return next_state, None

    return integrate_EM


def ode_integrator(obs: Float[Array, "obs"], diffusion_model, stop_grad: bool = False):
    """Probability-flow ODE of the controlled SDE: dX_t = (-1/2 sigma^2 X_t + 1/2 sigma u) dt.

    Replacing the Brownian term of the controlled SDE with a deterministic flow that
    preserves the marginals requires subtracting 1/2 sigma^2 * score. At optimality the
    control satisfies u = sigma * score of the controlled marginal, so the score term
    halves the control drift while the OU drift stays.

    Scan state: (x: Float[Array, "act"], key)."""
    dt = diffusion_model.dt

    def integrate_EM(state, step):
        step = step.astype(jnp.float32)
        x, key_gen = state
        t = step * dt

        sigma_t = diffusion_model.noise_scheduler.sigma_t(t)

        ctrl = diffusion_model.forward_model(t, x, obs)
        ou_drift = -0.5 * sigma_t ** 2 * x
        ctrl_drift = sigma_t * ctrl
        x_new = x + (ou_drift + 0.5 * ctrl_drift) * dt

        next_state = (x_new, key_gen)
        return next_state, None

    return integrate_EM

