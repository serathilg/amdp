import jax
import jax.numpy as jnp
from jaxtyping import Array, Float


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

        # Compute SDE components
        sigma_t = diffusion_model.noise_scheduler.sigma_t(t)

        # Update equation
        ctrl = sigma_t * diffusion_model.forward_model(t, x, obs)
        residual = sigma_t * ctrl * dt
        x_new = x + residual + sigma_t * noise * jnp.sqrt(dt)

        # Girsanov increments of log(dPi^0/dPi^u) between the uncontrolled reference
        # dX = sigma dB and the controlled dX = sigma ctrl dt + sigma dB (the
        # negated running cost and the negated stochastic integral).
        log_path_weight_deterministic += -0.5 * jnp.sum(ctrl ** 2) * dt
        log_path_weight_stochastic += -jnp.sum(ctrl * noise) * jnp.sqrt(dt)

        next_state = (x_new, log_path_weight_deterministic, log_path_weight_stochastic, key_gen)
        return next_state, None

    return integrate_EM
