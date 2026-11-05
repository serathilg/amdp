import torch
import math

def sde_integrator(obs, diffusion_model, stop_grad=False):
    dt = diffusion_model.dt
    
    def integrate_EM(x, log_path_weight_deterministic, log_path_weight_stochastic, step):
        # Sample standard normal noise with the same shape/device as x
        noise = torch.randn_like(x)
        t = step * dt

        # Compute SDE components
        sigma_t = diffusion_model.noise_schedule.sigma_t(t)

        ctrl = sigma_t * diffusion_model.forward_model(t, x, obs)
        
        if stop_grad:
            ctrl = ctrl.detach()
            
        residual = sigma_t * ctrl * dt
        x_new = x + residual + sigma_t * noise * math.sqrt(dt)

        # Sum over the action dimension (dim=-1) for the batch
        log_path_weight_deterministic = log_path_weight_deterministic - 0.5 * torch.sum(ctrl ** 2, dim=-1) * dt
        log_path_weight_stochastic = log_path_weight_stochastic - torch.sum(ctrl * noise, dim=-1) * math.sqrt(dt)

        return x_new, log_path_weight_deterministic, log_path_weight_stochastic

    return integrate_EM
