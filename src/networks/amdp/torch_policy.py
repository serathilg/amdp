import math
from typing import NamedTuple

import torch
from torch import nn


class SDESample(NamedTuple):
    """Per-sample statistics of one controlled-SDE action draw.

    Torch counterpart of the JAX SDESample with the same fields; unlike the JAX
    actor, the torch actor supports only the scaled-erf squashing.
    """

    action: torch.Tensor
    action_unsquashed: torch.Tensor
    prior_action: torch.Tensor
    squashing_correction_grad: torch.Tensor
    log_weight: torch.Tensor
    log_path_weight_deterministic: torch.Tensor
    log_path_weight_stochastic: torch.Tensor
    log_p_T_ref: torch.Tensor
    squashing_correction_val: torch.Tensor


class DiffusionModel(nn.Module):
    def __init__(
        self,
        action_dim: int,
        observation_dim: int,
        fwd_model: nn.Module = None,
        bwd_model: nn.Module = None,
        diff_steps: int = 8,
        noise_schedule = None,
        device=None,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.observation_dim = observation_dim
        self.diff_steps = diff_steps
        self.fwd_model = fwd_model
        self.bwd_model = bwd_model
        self.noise_schedule = noise_schedule
        assert self.noise_schedule is not None
        
        # The Brownian-reference SDE integrates with fixed 1/N step sizes
        self.dt = 1.0 / diff_steps

    def prior_sampler(self, n_samples, stop_grad, device=None):
        """Sample from the prior distribution (the Dirac prior at zero of the Brownian-reference SDE)."""
        samples = torch.zeros((n_samples, self.action_dim), device=device)
        return samples

    def prior_log_prob(self, x):
        """Compute log probability under the prior."""
        return self.ref_log_prob(x)

    def ref_log_prob(self, x):
        """Compute log probability under the reference MultivariateNormalDiag distribution."""
        std = self.noise_schedule.sigma_T_0()
            
        if not isinstance(std, torch.Tensor):
            std = torch.tensor(std, device=x.device, dtype=torch.float32)
        else:
            std = std.to(x.device)
            
        dist = torch.distributions.Independent(
            torch.distributions.Normal(torch.zeros_like(x), torch.ones_like(x) * std), 1
        )
        return dist.log_prob(x)

    def forward_model(self, step, x, obs):
        """Forward model function."""
        if self.fwd_model is not None:
            return self.fwd_model(x, obs, step)
        else:
            return torch.zeros_like(x)

    def backward_model(self, step, x, obs):
        """Backward model function."""
        if self.bwd_model is not None:
            return self.bwd_model(x, obs, step)
        else:
            return torch.zeros_like(x)


class DiffusionActor(nn.Module):
    def __init__(
        self,
        action_dim: int,
        observation_dim: int,
        diffusion_model: nn.Module,
        sde_integrator: callable = None,
        kl_start: float = 0.1,
        ent_start: float = 0.1,
        kl_bound: float = 0.1,
        asymmetric_obs: bool = False,
        device=None,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.observation_dim = observation_dim
        self.diffusion_model = diffusion_model
        self.sde_integrator = sde_integrator

        self.log_temperature = nn.Parameter(
            torch.ones(1, device=device) * math.log(ent_start)
        )
        self.log_lagrangian = nn.Parameter(
            torch.ones(1, device=device) * math.log(kl_start)
        )
        
        # Setup Squashing Constant for Erf transformation
        if hasattr(self.diffusion_model, "noise_schedule") and self.diffusion_model.noise_schedule is not None:
            sigma_T_0 = self.diffusion_model.noise_schedule.sigma_T_0()
        else:
            sigma_T_0 = self.diffusion_model.init_std
            
        if isinstance(sigma_T_0, torch.Tensor):
            sigma_T_0 = sigma_T_0.item()
            
        self.squash_k = math.sqrt((sigma_T_0**-2) / 2.0)
    
    def sde_sample(
        self,
        obs: torch.Tensor,
        stop_grad: bool = False,
    ) -> SDESample:
        """Sample actions stochastically via the SDE."""
        bs, *_ = obs.shape
        init_x = self.diffusion_model.prior_sampler(bs, stop_grad=stop_grad, device=obs.device)

        integrator_fn = self.sde_integrator(obs, self.diffusion_model, stop_grad=stop_grad)

        log_path_weight_deterministic = torch.zeros(bs, device=obs.device, dtype=torch.float32)
        log_path_weight_stochastic = torch.zeros(bs, device=obs.device, dtype=torch.float32)
        x = init_x

        for step in range(self.diffusion_model.diff_steps):
            step_t = torch.tensor(step, dtype=torch.float32, device=obs.device)
            x, log_path_weight_deterministic, log_path_weight_stochastic = integrator_fn(
                x, log_path_weight_deterministic, log_path_weight_stochastic, step_t
            )

        final_x_unsquashed = x
        final_action = torch.erf(self.squash_k * final_x_unsquashed)
        
        log_p_T_ref = self.diffusion_model.ref_log_prob(final_x_unsquashed)
        
        # log|det J_f| of the scaled-erf squashing, per dim:
        # log|f'(x)| = log(2k) - 0.5*log(pi) - (kx)^2
        # sum over action dim
        squashing_correction_val = (math.log(2 * self.squash_k) - 0.5 * math.log(math.pi) - (self.squash_k * final_x_unsquashed)**2).sum(dim=-1)
        
        # nabla log|f'(x)| = - 2 k^2 x
        squashing_correction_grad = -2 * (self.squash_k**2) * final_x_unsquashed
        
        log_weight = log_path_weight_deterministic + log_path_weight_stochastic - log_p_T_ref + squashing_correction_val

        return SDESample(
            action=final_action,
            action_unsquashed=final_x_unsquashed,
            prior_action=init_x,
            squashing_correction_grad=squashing_correction_grad,
            log_weight=log_weight,
            log_path_weight_deterministic=log_path_weight_deterministic,
            log_path_weight_stochastic=log_path_weight_stochastic,
            log_p_T_ref=log_p_T_ref,
            squashing_correction_val=squashing_correction_val,
        )

    def forward(self, obs: torch.Tensor, stop_grad: bool = False) -> tuple:
        """Forward pass - sample actions from diffusion model."""
        ret = self.sde_sample(obs, stop_grad=stop_grad)
        return ret.action, ret.log_path_weight_deterministic, ret.log_path_weight_stochastic, ret.log_p_T_ref

    def temperature(self) -> torch.Tensor:
        """Get current dynamic temperature value."""
        return torch.exp(self.log_temperature)
    
    def lagrangian(self) -> torch.Tensor:
        """Get current lagrangian multiplier value."""
        return torch.exp(self.log_lagrangian)