import torch
import math
from typing import NamedTuple, Callable

class NoiseScheduler(NamedTuple):
    sigma_t: Callable
    sigma_t_0: Callable
    sigma_T_0: Callable
    sigma_t_T: Callable
    sigma_t_0T: Callable
    mu_t_0T_scale: Callable
    xT_0T_scale: Callable
    x0_0T_scale: Callable
    noise_0T_scale: Callable

def get_geometric_scheduler(sigma_max, sigma_min):
    r = sigma_max / sigma_min
    log_r = math.log(r)

    def sigma_t(t):
        return sigma_min * (r ** (1 - t)) * math.sqrt(2 * log_r)

    def sigma_t_0(t):
        return torch.sqrt(sigma_max**2 * (1 - r ** (-2 * t)))

    sigma_T_0_ = math.sqrt(sigma_max**2 * (1 - r ** (-2.0)))

    def sigma_T_0():
        return sigma_T_0_

    def sigma_t_T(t):
        return torch.sqrt(sigma_T_0_ ** 2 - sigma_t_0(t) ** 2)

    def sigma_t_0T(t):
        sigma_t_0_val = sigma_t_0(t)
        sigma_ratio = sigma_t_0_val / sigma_T_0_
        return sigma_t_0_val * torch.sqrt(1 - sigma_ratio**2)

    def mu_t_0T_scale(t):
        return sigma_t_0(t) ** 2 / sigma_T_0_ ** 2

    def xT_0T_scale(t):
        return sigma_t_0(t) ** 2 / sigma_T_0_ ** 2

    def x0_0T_scale(t):
        return 1 - xT_0T_scale(t)

    def noise_0T_scale(t):
        sigma_t_0_val = sigma_t_0(t) ** 2
        return torch.sqrt(sigma_t_0_val * (sigma_T_0_ ** 2 - sigma_t_0_val) / sigma_T_0_ ** 2)

    return NoiseScheduler(
        sigma_t=sigma_t, sigma_t_0=sigma_t_0, sigma_T_0=sigma_T_0,
        sigma_t_T=sigma_t_T, sigma_t_0T=sigma_t_0T, mu_t_0T_scale=mu_t_0T_scale,
        xT_0T_scale=xT_0T_scale, x0_0T_scale=x0_0T_scale, noise_0T_scale=noise_0T_scale,
    )