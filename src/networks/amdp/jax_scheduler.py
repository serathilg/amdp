from typing import NamedTuple, Callable

import jax.numpy as jnp
from jaxtyping import Array, Float

# Every schedule callable maps a diffusion time t in [0, 1] to a value of the
# same shape; the annotations are documentation, not runtime-checked.
ScheduleFn = Callable[[Float[Array, "*batch"] | float], Float[Array, "*batch"]]


class NoiseScheduler(NamedTuple):
    """Noise-schedule callables; each maps t (float or Float[Array, "*batch"]
    in [0, 1]) to the same shape. sigma_T_0 takes no argument."""

    sigma_t: ScheduleFn
    sigma_t_0: ScheduleFn
    sigma_T_0: Callable[[], Float[Array, ""]]
    sigma_t_T: ScheduleFn
    sigma_t_0T: ScheduleFn
    mu_t_0T_scale: ScheduleFn
    xT_0T_scale: ScheduleFn
    x0_0T_scale: ScheduleFn
    noise_0T_scale: ScheduleFn


def get_geometric_scheduler(sigma_max: float, sigma_min: float) -> NoiseScheduler:
    sigma_min = sigma_min
    sigma_max = sigma_max
    r = sigma_max / sigma_min
    log_r = jnp.log(r)

    def sigma_t(t):
        return sigma_min * r ** (1 - t) * jnp.sqrt(2 * log_r)

    def sigma_t_0(t):  # cumulative std from t=0 to t
        return jnp.sqrt(sigma_max**2 * (1 - r ** (-2 * t)))

    sigma_T_0_ = sigma_t_0(1.0)

    def sigma_T_0():
        return sigma_T_0_

    def sigma_t_T(t):
        return jnp.sqrt(sigma_t_0(1.0) ** 2 - sigma_t_0(t) ** 2)

    def sigma_t_0T(t):  # STD at time step t given 0 and T
        sigma_t_0_ = sigma_t_0(t)
        sigma_ratio = sigma_t_0_ / sigma_T_0_
        return sigma_t_0_ * jnp.sqrt(1 - sigma_ratio**2)

    def mu_t_0T_scale(t):  # Mean von p(t|0,T)
        return sigma_t_0(t) ** 2 / sigma_T_0() ** 2

    def xT_0T_scale(t):
        return sigma_t_0(t) ** 2 / sigma_T_0() ** 2

    def x0_0T_scale(t):
        return 1 - xT_0T_scale(t)

    def noise_0T_scale(t):
        sigma_t_0_val = sigma_t_0(t) ** 2
        return jnp.sqrt(sigma_t_0_val * (sigma_T_0() ** 2 - sigma_t_0_val) / sigma_T_0() ** 2)

    return NoiseScheduler(
        sigma_t=sigma_t,
        sigma_t_0=sigma_t_0,
        sigma_T_0=sigma_T_0,
        sigma_t_T=sigma_t_T,
        sigma_t_0T=sigma_t_0T,
        mu_t_0T_scale=mu_t_0T_scale,
        xT_0T_scale=xT_0T_scale,
        x0_0T_scale=x0_0T_scale,
        noise_0T_scale=noise_0T_scale,
    )
