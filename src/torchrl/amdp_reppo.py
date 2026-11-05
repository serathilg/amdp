from dataclasses import dataclass, replace
import functools
import os
import random
import shutil
import sys
import copy
import time
import math
from pathlib import Path

import numpy as np
import tqdm
from omegaconf import DictConfig, OmegaConf

import wandb

from src.torchrl.reppo_util import EmpiricalNormalization, hl_gauss
from src.torchrl.trajectory_utils import save_and_plot_trajectories
from src.torchrl.env_hyperparams import apply_env_specific_hyperparams

import hydra
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchinfo import summary
from tensordict import TensorDict
from torch.amp import GradScaler
from src.torchrl.envs import make_envs
from src.networks.torch_models import FCNN, Critic

from src.networks.amdp.torch_policy import DiffusionModel, DiffusionActor, SDESample
from src.networks.amdp.torch_control_net import ControlNetwork


torch.set_float32_matmul_precision("high")
os.environ["TORCHDYNAMO_INLINE_INBUILT_NN_MODULES"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
if sys.platform != "darwin":
    os.environ["MUJOCO_GL"] = "egl"
else:
    os.environ["MUJOCO_GL"] = "glfw"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["JAX_DEFAULT_MATMUL_PRECISION"] = "highest"


@dataclass(slots=True)
class TrainState:
    device: torch.device
    obs: torch.Tensor
    critic_obs: torch.Tensor
    actor: DiffusionActor
    old_actor: DiffusionActor
    critic: Critic
    critic_target: Critic
    normalizer: EmpiricalNormalization
    critic_normalizer: EmpiricalNormalization
    actor_optimizer: optim.Optimizer
    critic_optimizer: optim.Optimizer
    scaler: GradScaler

    def compile(self):
        self.actor.compile()
        self.old_actor.compile()
        self.critic.compile()
        self.critic_target.compile()
        self.normalizer.compile()
        self.critic_normalizer.compile()


def get_autocast_context(cfg: DictConfig):
    amp_enabled = cfg.platform.amp_enabled and cfg.platform.cuda and torch.cuda.is_available()
    amp_device = (
        "cuda"
        if cfg.platform.cuda and torch.cuda.is_available()
        else "mps" if cfg.platform.cuda and torch.backends.mps.is_available() else "cpu"
    )
    amp_dtype = torch.bfloat16 if cfg.platform.amp_dtype == "bf16" else torch.float32
    return functools.partial(
        torch.amp.autocast,
        device_type=amp_device,
        dtype=amp_dtype,
        enabled=amp_enabled,
    )


def create_staggered_resets(max_episode_steps: int, num_envs: int, num_groups_requested: int, device: torch.device):
    """
    Calculates initial elapsed_step offsets to desynchronize environment resets.
    """
    if num_envs == 0:
        return torch.empty(0, dtype=torch.long, device=device)

    # Determine valid number of groups
    if num_groups_requested is None or num_groups_requested <= 0:
        actual_num_groups = 1
    elif num_groups_requested > num_envs:
        actual_num_groups = num_envs
    else:
        actual_num_groups = num_groups_requested

    # Assign envs to groups
    envs_per_group_approx = max(1, num_envs // actual_num_groups)
    group_indices = torch.arange(num_envs, device=device) // envs_per_group_approx
    group_indices = torch.clamp(group_indices, 0, actual_num_groups - 1)

    # Calculate offset step size
    step_size = max_episode_steps // actual_num_groups

    # Calculate starting elapsed steps
    elapsed_steps = group_indices * step_size

    return elapsed_steps


def make_collect_fn(cfg: DictConfig, env, env_type: str):
    autocast = get_autocast_context(cfg)
    asymmetric_obs = env.asymmetric_obs

    def collect_fn(
        train_state: TrainState,
    ) -> tuple[TrainState, TensorDict, list[dict]]:
        transitions = []
        info_list = []
        obs = train_state.obs
        critic_obs = train_state.critic_obs

        # we merge the next state entropy action sample with the next action
        # by shifting when to loop. This needs an initial action sample.
        with autocast(), torch.inference_mode():
            norm_obs = train_state.normalizer(obs)
            norm_critic_obs = train_state.critic_normalizer(critic_obs)
            sample = train_state.actor.sde_sample(norm_obs)

        for _ in range(cfg.hyperparameters.num_steps):
            if env_type == "maniskill":
                next_obs, rewards, dones, truncations, infos = env.step(sample.action)
            elif env_type == "humanoid_bench":
                next_obs, rewards, dones, infos = env.step(sample.action)
                truncations = infos["time_outs"]
            else:
                next_obs, rewards, dones, truncations, infos = env.step(sample.action)

            if asymmetric_obs:
                next_critic_obs = infos["observations"]["critic"]
            else:
                next_critic_obs = next_obs

            # Handle final observations
            if (
                cfg.env.get("has_final_obs", False)
                and cfg.env.get("partial_reset", False)
                and "final_observation" in infos
            ):
                _next_obs = infos["final_observation"]
                _next_critic_obs = _next_obs
            else:
                _next_obs = next_obs
                _next_critic_obs = next_critic_obs

            with torch.inference_mode(), autocast():
                # Stack _next_obs (entropy for current step, possibly beyond truncation)
                # and next_obs (action for next step) into shape [2, B, ...]
                combined_obs = torch.cat([_next_obs, next_obs], dim=0)
                norm_combined_obs = train_state.normalizer(combined_obs)
                temperature = train_state.actor.temperature()

                combined_sample = train_state.actor.sde_sample(norm_combined_obs, stop_grad=True)
                target_sample = SDESample(*(t.chunk(2, dim=0)[0] for t in combined_sample))
                next_step_sample = SDESample(*(t.chunk(2, dim=0)[1] for t in combined_sample))

                next_log_probs = -target_sample.log_weight
                next_actions = target_sample.action

                norm_next_critic_obs = train_state.critic_normalizer(_next_critic_obs)
                next_value, _, _, next_embedding = train_state.critic_target(norm_next_critic_obs, next_actions)

                rewards = rewards - cfg.hyperparameters.gamma * next_log_probs.squeeze(-1) * temperature

            transitions.append(
                TensorDict(
                    {
                        "observations": norm_obs,
                        "critic_observations": norm_critic_obs,
                        "actions": sample.action,
                        "actions_unsquashed": sample.action_unsquashed,
                        "prior_actions": sample.prior_action,
                        "squashing_correction_grads": sample.squashing_correction_grad,
                        "squashing_correction_vals": sample.squashing_correction_val,
                        "log_weights": sample.log_weight,
                        "log_p_T_ref": sample.log_p_T_ref,
                        "rewards": rewards.unsqueeze(-1),
                        "next_embeddings": next_embedding,
                        "next_values": next_value.unsqueeze(-1),
                        "dones": dones.unsqueeze(-1).float(),
                        "truncations": truncations.unsqueeze(-1).float(),
                        "Q_values": torch.zeros_like(next_value).unsqueeze(-1),
                        "Q_scores": torch.zeros_like(sample.action),
                    },
                    batch_size=(env.num_envs,),
                )
            )
            info_list.append(infos)

            # offset step transition here to allow for action sampling stacking
            obs = next_obs
            critic_obs = next_critic_obs
            norm_obs = norm_combined_obs.chunk(2, dim=0)[1]  # Already normalized
            norm_critic_obs = train_state.critic_normalizer(critic_obs)

            sample = next_step_sample

        train_state = replace(train_state, obs=obs, critic_obs=critic_obs)
        return train_state, torch.stack(transitions, dim=0), info_list

    return collect_fn


def make_postprocess_fn(cfg: DictConfig, env):
    @torch.compiler.disable()
    def compute_gve(rewards, dones, truncated, next_values, device: torch.device):
        gves = []
        last_gve = 0
        truncated[-1] = 1.0
        for t in reversed(range(cfg.hyperparameters.num_steps)):
            lambda_sum = cfg.hyperparameters.lmbda * last_gve + (1.0 - cfg.hyperparameters.lmbda) * next_values[t]
            delta = cfg.hyperparameters.gamma * torch.where(
                truncated[t].bool(), next_values[t], (1.0 - dones[t]) * lambda_sum
            )
            last_gve = rewards[t] + delta
            gves.insert(0, last_gve)
        return gves

    def postprocess(train_state: TrainState, transition: TensorDict):
        gve = compute_gve(
            rewards=transition["rewards"],
            dones=transition["dones"],
            truncated=transition["truncations"],
            next_values=transition["next_values"],
            device=train_state.device,
        )

        data = TensorDict(
            {
                "observations": transition["observations"],
                "critic_observations": transition["critic_observations"],
                "actions": transition["actions"],
                "actions_unsquashed": transition["actions_unsquashed"],
                "prior_actions": transition["prior_actions"],
                "squashing_correction_grads": transition["squashing_correction_grads"],
                "squashing_correction_vals": transition["squashing_correction_vals"],
                "log_weights": transition["log_weights"],
                "log_p_T_ref": transition["log_p_T_ref"],
                "rewards": transition["rewards"],
                "next_embeddings": transition["next_embeddings"],
                "next_values": transition["next_values"],
                "dones": transition["dones"],
                "truncations": transition["truncations"],
                "Q_values": transition["Q_values"],
                "Q_scores": transition["Q_scores"],
                "gve": torch.stack(gve),
            },
            batch_size=(
                cfg.hyperparameters.num_steps,
                cfg.hyperparameters.num_envs,
            ),
            device=train_state.device,
        )
        return data.float().flatten(0, 1).detach()

    return postprocess


def update_Q(data: TensorDict, train_state: TrainState, cfg: DictConfig):
    """Computes nabla_a Q(s,a) w.r.t unsquashed actions and applies percentile clipping."""
    obs = data["critic_observations"].detach().requires_grad_(True)
    action_unsq = data["actions_unsquashed"].detach().requires_grad_(True)

    schedule = train_state.actor.diffusion_model.noise_schedule
    sigma_P_T = schedule.sigma_T_0()

    k_val = math.sqrt((sigma_P_T**-2) / 2.0)
    action_squashed = torch.erf(k_val * action_unsq)

    q_val, _, _, _ = train_state.critic_target(obs, action_squashed)

    q_scores = torch.autograd.grad(
        outputs=q_val, inputs=action_unsq, grad_outputs=torch.ones_like(q_val), retain_graph=False
    )[0]

    norms = q_scores.norm(dim=-1, keepdim=True)

    percentile = cfg.hyperparameters.get("Q_score_max_percentile", 95.0) / 100.0
    batch_p95 = torch.quantile(norms.float(), percentile).to(norms.dtype)

    max_norm = cfg.hyperparameters.get("Q_score_max_norm", 1.0)
    clip_thresh = torch.minimum(batch_p95, torch.tensor(max_norm, device=norms.device, dtype=norms.dtype))

    scale_factor = torch.where(norms > clip_thresh, clip_thresh / (norms + 1e-6), torch.ones_like(norms))

    q_scores_clipped = q_scores * scale_factor

    data["Q_values"] = q_val.detach()
    data["Q_scores"] = q_scores_clipped.detach()
    return data


def make_critic_update_fn(cfg: DictConfig, train_state: TrainState):
    autocast = get_autocast_context(cfg)

    # avoid frequent dict lookup in update calls
    partial_reset = cfg.env.get("partial_reset", False)
    vmin = cfg.hyperparameters.vmin
    vmax = cfg.hyperparameters.vmax
    num_bins = cfg.hyperparameters.num_bins
    aux_loss_mult = cfg.hyperparameters.aux_loss_mult
    max_grad_norm = cfg.hyperparameters.max_grad_norm

    def update(data: TensorDict):
        qnet = train_state.critic
        q_optimizer = train_state.critic_optimizer

        with autocast():
            critic_observations = data["critic_observations"]
            actions = data["actions"]
            targets = data["gve"]
            target_embeddings = data["next_embeddings"]
            truncations = data["truncations"].squeeze(-1)
            if partial_reset:
                truncation_mask = torch.ones_like(truncations, dtype=torch.bool, device=train_state.device)
            else:
                truncation_mask = 1.0 - truncations
            qf_target_dist = hl_gauss(
                targets,
                vmin,
                vmax,
                num_bins,
            )

            _, qf1, embedding, _ = qnet(critic_observations, actions)
            qf_loss = -(truncation_mask * torch.sum(qf_target_dist * F.log_softmax(qf1, dim=-1), dim=-1)).mean()
            embedding_loss = (
                truncation_mask
                * F.mse_loss(
                    embedding,
                    target_embeddings,
                    reduction="none",
                ).mean(dim=-1)
            ).mean()

            qf_loss = qf_loss + aux_loss_mult * embedding_loss

        q_optimizer.zero_grad(set_to_none=True)
        train_state.scaler.scale(qf_loss).backward()
        train_state.scaler.unscale_(q_optimizer)

        critic_grad_norm = torch.nn.utils.clip_grad_norm_(qnet.parameters(), max_norm=max_grad_norm)
        train_state.scaler.step(q_optimizer)
        train_state.scaler.update()
        logs_dict = {
            "critic_grad_norm": critic_grad_norm.detach(),
            "qf_loss": qf_loss.detach(),
            "qf_max": targets.max().detach(),
            "qf_min": targets.min().detach(),
            "qf_mean": targets.mean().detach(),
            "embedding_loss": embedding_loss.detach(),
        }
        return logs_dict

    return update


def compute_entropy_via_importance_sampling(log_weights, Q_value, lm_entr):
    r"""SNIS estimate of the entropy of the optimal policy pi*(a|s) \propto
    exp(Q(a,s)/T), the exp(Q/T)-tilting of the squashed reference. The weight
    log w = Q/T + log_weights = log dPi*/dPi^u(X) + log Z is the path RND of
    the terminally tilted reference vs the policy; see the derivation in
    src/jaxrl/lagrangian_utils.py."""
    T = lm_entr
    log_q_tilde = Q_value / T
    log_importance_weights = log_q_tilde + log_weights
    N = log_q_tilde.shape[-1]

    log_Z = torch.logsumexp(log_importance_weights, dim=-1) - math.log(N)
    norm_weights = torch.nn.functional.softmax(log_importance_weights, dim=-1)

    entropy = -torch.sum(norm_weights * log_q_tilde, dim=-1) + log_Z
    return entropy


def parallel_nary_search(f, low, high, n_points=256, rtol=1e-4, atol=1e-6, max_iter=8, device=None):
    """
    Parallel N-ary search of convex 1d function of positive input with expansion of range.
    """
    assert low > 0 and high > 0, f"low and high of range must be positive, got {low=} and {high=}"

    a = torch.tensor(low, dtype=torch.float32, device=device)
    b = torch.tensor(high, dtype=torch.float32, device=device)
    i = 0

    while i < max_iter and (b / a) > (1.0 + rtol) and torch.abs(b - a) > atol:
        grid = torch.logspace(torch.log10(a), torch.log10(b), n_points, base=10.0, device=device)

        vals = f(grid)
        idx = torch.argmin(vals)

        ratio = b / a

        is_at_lower = idx == 0
        high_lower = grid[1]
        low_lower = high_lower / ratio

        is_at_upper = idx == n_points - 1
        low_upper = grid[n_points - 2]
        high_upper = low_upper * ratio

        idx_down = torch.clamp(idx - 1, min=0)
        idx_up = torch.clamp(idx + 1, max=n_points - 1)
        low_bracketing = grid[idx_down]
        high_bracketing = grid[idx_up]

        if is_at_lower:
            new_low, new_high = low_lower, high_lower
        elif is_at_upper:
            new_low, new_high = low_upper, high_upper
        else:
            new_low, new_high = low_bracketing, high_bracketing

        a, b = new_low, new_high
        i += 1

    return (a + b) / 2.0


def find_optimum_kl_lagrangian(w_t, old_ctrl, ctrl_target, eps, min_val=1e-3, max_val=1e6, norm_weights=True):
    if norm_weights:
        w_t_factor = w_t.mean()
    else:
        w_t_factor = 1.0

    w_t_norm = w_t / w_t_factor
    sse = 0.5 * torch.sum((old_ctrl - ctrl_target) ** 2, dim=-1)
    w2 = w_t_norm**2

    def eval_single_lambda(lam):
        # lam shape: (n_points,) -> unsqueeze to (n_points, 1)
        # w_t_norm shape: (B,) -> unsqueeze to (1, B)
        w_lam_2 = (w_t_norm.unsqueeze(0) + lam.unsqueeze(1)) ** 2
        dual_grad = torch.mean(w2.unsqueeze(0) / w_lam_2 * sse.unsqueeze(0), dim=1) - eps
        return dual_grad**2

    min_norm_lambda = parallel_nary_search(eval_single_lambda, low=min_val, high=max_val, device=w_t.device)
    return min_norm_lambda * w_t_factor


def make_actor_update_fn(cfg: DictConfig, train_state: TrainState):
    autocast = get_autocast_context(cfg)
    kl_bound = cfg.hyperparameters.kl_bound
    ent_target_mult = cfg.hyperparameters.ent_target_mult
    max_grad_norm = cfg.hyperparameters.max_grad_norm

    def update(data: TensorDict):
        actor = train_state.actor
        old_actor = train_state.old_actor
        actor_optimizer = train_state.actor_optimizer
        scaler = train_state.scaler
        noise_schedule = actor.diffusion_model.noise_schedule

        obs = data["observations"]
        a_T_unsq = data["actions_unsquashed"]
        Q_score = data["Q_scores"]
        Q_value = data["Q_values"]
        log_weights = data["log_weights"]
        squashing_correction_grad = data["squashing_correction_grads"]

        batch_size = a_T_unsq.shape[0]

        with autocast():
            t = torch.rand((batch_size, 1), device=train_state.device)
            noise = torch.randn_like(a_T_unsq)

            mu_scale = noise_schedule.mu_t_0T_scale(t)
            sigma_scale = noise_schedule.sigma_t_0T(t)
            sigma_t = noise_schedule.sigma_t(t)
            sigma_T_0 = noise_schedule.sigma_T_0()

            a_t = mu_scale * a_T_unsq + sigma_scale * noise

            ctrl = sigma_t * actor.diffusion_model.fwd_model(a_t, obs, t)
            with torch.no_grad():
                old_ctrl = sigma_t * old_actor.diffusion_model.fwd_model(a_t, obs, t)

            temperature = actor.temperature()
            temp_scaler = temperature.detach()

            nabla_p_T_ref = -a_T_unsq / (sigma_T_0**2)
            adjoint_state = (nabla_p_T_ref - squashing_correction_grad) - (Q_score / temp_scaler)
            ctrl_target = -sigma_t * adjoint_state

            adjoint_loss_unweighted = 0.5 * torch.sum((ctrl - ctrl_target) ** 2, dim=-1)

            # 1/sigma_t loss weighting
            w_t = sigma_t.squeeze(-1) ** -1

            adjoint_loss = (adjoint_loss_unweighted * w_t).mean()

            opt_lm = find_optimum_kl_lagrangian(w_t, old_ctrl, ctrl_target, eps=kl_bound)
            lagrangian = actor.lagrangian() * opt_lm.detach()
            kl_scale = lagrangian.detach()

            kl_loss = 0.5 * torch.sum((ctrl - old_ctrl) ** 2, dim=-1)
            scaled_kl_loss = kl_scale * kl_loss.mean()

            actor_loss = adjoint_loss + scaled_kl_loss

            target_entropy = a_T_unsq.shape[-1] * ent_target_mult
            entropy_loss = temperature * (target_entropy + log_weights.mean()).detach()

            optimal_entropy = compute_entropy_via_importance_sampling(
                log_weights.reshape(-1), Q_value.reshape(-1), temp_scaler
            )

            lagrangian_loss = -lagrangian * (kl_loss.mean() - kl_bound).detach()

            total_loss = actor_loss + entropy_loss + lagrangian_loss

        actor_optimizer.zero_grad(set_to_none=True)
        scaler.scale(total_loss).backward()
        scaler.unscale_(actor_optimizer)
        actor_grad_norm = torch.nn.utils.clip_grad_norm_(actor.parameters(), max_norm=max_grad_norm)
        scaler.step(actor_optimizer)
        scaler.update()

        return {
            "actor_loss": actor_loss.detach(),
            "adjoint_loss": adjoint_loss.detach(),
            "kl_loss": kl_loss.mean().detach(),
            "actor_grad_norm": actor_grad_norm.detach(),
            "entropy": log_weights.mean().detach(),
            "optimal_entropy": optimal_entropy.detach(),
            "temperature": temperature.detach(),
            "lagrangian": lagrangian.detach(),
            "dual_lm": actor.lagrangian().detach(),
            "opt_lm": opt_lm.detach(),
        }

    return update


def make_sde_eval_fn(cfg: DictConfig, eval_envs, env_type: str):
    """
    Creates evaluation function for SDE (stochastic) policy.
    """
    autocast = get_autocast_context(cfg)

    @torch.inference_mode()
    def sde_evaluate(train_state: TrainState) -> dict[str, float]:
        train_state.normalizer.eval()
        num_eval_envs = eval_envs.num_envs
        episode_returns = torch.zeros(num_eval_envs, device=train_state.device)
        episode_lengths = torch.zeros(num_eval_envs, device=train_state.device)
        done_masks = torch.zeros(num_eval_envs, dtype=torch.bool, device=train_state.device)

        if cfg.env.asymmetric_obs:
            obs, critic_obs = eval_envs.reset_with_critic_obs()
        else:
            if cfg.env.type == "maniskill":
                (obs, *_) = eval_envs.reset()
            else:
                obs = eval_envs.reset()

        for i in range(eval_envs.max_episode_steps):
            with autocast():
                obs_normalized = train_state.normalizer(obs)
            actions = train_state.actor.sde_sample(obs_normalized).action

            if env_type == "maniskill":
                next_obs, rewards, dones, _, infos = eval_envs.step(actions)
            elif env_type == "humanoid_bench":
                next_obs, rewards, dones, infos = eval_envs.step(actions)
            else:
                next_obs, rewards, dones, _, infos = eval_envs.step(actions)

            episode_returns = torch.where(~done_masks, episode_returns + rewards, episode_returns)
            episode_lengths = torch.where(~done_masks, episode_lengths + 1, episode_lengths)
            done_masks = torch.logical_or(done_masks, dones)
            if done_masks.all():
                break
            obs = next_obs

        train_state.normalizer.train()

        result = {
            "episode_return": episode_returns.mean().item(),
            "episode_length": episode_lengths.mean().item(),
        }

        if cfg.env.type == "maniskill":
            result.update(
                {
                    "info_return": infos["log_info"]["return"].mean().item(),
                    "success": infos["log_info"]["success"].float().mean().item(),
                }
            )

        return result

    return sde_evaluate


def make_best_of_n_eval_fn(cfg: DictConfig, eval_envs, env_type: str):
    """
    Creates evaluation function for Best-of-N sampling: N SDE samples are drawn
    per step and the action with the highest Q-value is executed.
    """
    autocast = get_autocast_context(cfg)

    @torch.inference_mode()
    def best_of_n_evaluate(train_state: TrainState, best_of_n: float = 1.0) -> dict[str, float]:
        train_state.normalizer.eval()
        num_eval_envs = eval_envs.num_envs
        episode_returns = torch.zeros(num_eval_envs, device=train_state.device)
        episode_lengths = torch.zeros(num_eval_envs, device=train_state.device)
        done_masks = torch.zeros(num_eval_envs, dtype=torch.bool, device=train_state.device)

        if cfg.env.asymmetric_obs:
            obs, critic_obs = eval_envs.reset_with_critic_obs()
        else:
            if cfg.env.type == "maniskill":
                (obs, *_) = eval_envs.reset()
            else:
                obs = eval_envs.reset()

        for i in range(eval_envs.max_episode_steps):
            with autocast():
                obs_normalized = train_state.normalizer(obs)
            BoN = int(best_of_n)
            obs_shape = obs_normalized.shape
            repeated_obs = obs_normalized.unsqueeze(0).expand(BoN, *obs_shape)
            repeated_obs = repeated_obs.reshape(BoN * obs_shape[0], *obs_shape[1:])
            actions = train_state.actor.sde_sample(repeated_obs).action
            q_vals, *_ = train_state.critic(repeated_obs, actions)
            actions_per_obs = actions.view(BoN, obs_shape[0], *actions.shape[1:])
            qs_per_obs = q_vals.view(BoN, obs_shape[0], *q_vals.shape[1:])
            actions = actions_per_obs[qs_per_obs.argmax(axis=0), torch.arange(obs_shape[0])]

            if env_type == "maniskill":
                next_obs, rewards, dones, _, infos = eval_envs.step(actions)
            elif env_type == "humanoid_bench":
                next_obs, rewards, dones, infos = eval_envs.step(actions)
            else:
                next_obs, rewards, dones, _, infos = eval_envs.step(actions)

            episode_returns = torch.where(~done_masks, episode_returns + rewards, episode_returns)
            episode_lengths = torch.where(~done_masks, episode_lengths + 1, episode_lengths)
            done_masks = torch.logical_or(done_masks, dones)
            if done_masks.all():
                break
            obs = next_obs

        train_state.normalizer.train()

        result = {
            "episode_return": episode_returns.mean().item(),
            "episode_length": episode_lengths.mean().item(),
        }

        if cfg.env.type == "maniskill":
            result.update(
                {
                    "info_return": infos["log_info"]["return"].mean().item(),
                    "success": infos["log_info"]["success"].float().mean().item(),
                }
            )

        return result

    return best_of_n_evaluate


def make_evaluate_fn(cfg: DictConfig, eval_envs, env_type: str):
    """
    Creates a unified evaluation function that supports both SDE and ODE modes.
    For backward compatibility with existing code.
    """
    autocast = get_autocast_context(cfg)

    @torch.inference_mode()
    def evaluate(
        train_state: TrainState
    ) -> tuple[int | float | bool, int | float | bool]:
        train_state.normalizer.eval()
        num_eval_envs = eval_envs.num_envs
        episode_returns = torch.zeros(num_eval_envs, device=train_state.device)
        episode_lengths = torch.zeros(num_eval_envs, device=train_state.device)
        done_masks = torch.zeros(num_eval_envs, dtype=torch.bool, device=train_state.device)

        if cfg.env.asymmetric_obs:
            obs, critic_obs = eval_envs.reset_with_critic_obs()
        else:
            if cfg.env.type == "maniskill":
                (obs, *_) = eval_envs.reset()
                critic_obs = obs
            else:
                obs = eval_envs.reset()
                critic_obs = obs

        for i in range(eval_envs.max_episode_steps):
            with autocast():
                obs_norm = train_state.normalizer(obs)
            actions = train_state.actor.sde_sample(obs_norm).action

            if env_type == "maniskill":
                next_obs, rewards, dones, _, infos = eval_envs.step(actions)
            elif env_type == "humanoid_bench":
                next_obs, rewards, dones, infos = eval_envs.step(actions)
                truncations = infos["time_outs"]
            else:
                next_obs, rewards, dones, _, infos = eval_envs.step(actions)

            episode_returns = torch.where(~done_masks, episode_returns + rewards, episode_returns)
            episode_lengths = torch.where(~done_masks, episode_lengths + 1, episode_lengths)
            done_masks = torch.logical_or(done_masks, dones)
            if done_masks.all():
                break
            obs = next_obs

        train_state.normalizer.train()

        if cfg.env.type == "maniskill":
            info = {
                "info_return": infos["log_info"]["return"].mean(),
                "episode_len": infos["log_info"]["episode_len"].float().mean(),
                "success": infos["log_info"]["success"].float().mean(),
                "return": episode_returns.mean().item(),
            }
        else:
            info = {}

        return episode_returns.mean().item(), episode_lengths.mean().item(), info

    return evaluate


def make_render_fn(cfg: DictConfig, render_env, env_type: str):
    """Create a render function for video recording and trajectory tracking."""
    autocast = get_autocast_context(cfg)

    def render_with_rollout(train_state: TrainState, track_trajectory: bool = False):
        """Perform a rollout and record video frames. Optionally track end-effector trajectories."""
        train_state.normalizer.eval()
        trajectories = [] if track_trajectory else None

        if env_type == "humanoid_bench":
            obs = render_env.reset()
            renders = [render_env.render()]
        elif env_type == "maniskill":
            # For ManiSkill, use the built-in recording capabilities
            (obs, *_) = render_env.reset(seed=1)
            renders = []
            has_recording = hasattr(render_env, "unwrapped") and hasattr(render_env.unwrapped, "_record_episode")
            if has_recording:
                print("Using ManiSkill built-in recording")
        elif env_type == "mjx":
            obs = render_env.reset()
            try:
                import jax.numpy as jnp

                render_env.state.info["command"] = jnp.array([[1.0, 0.0, 0.0]])
            except (ImportError, AttributeError):
                pass
            renders = [render_env.state]
        else:
            obs = render_env.reset()
            renders = []

        max_steps = getattr(render_env, "max_episode_steps", 1000)
        episode_frames = []

        for i in range(max_steps):
            with torch.no_grad(), autocast():
                obs_norm = train_state.normalizer(obs)
                actions = train_state.actor.sde_sample(obs_norm).action

            if env_type == "maniskill":
                next_obs, rewards, dones, _, infos = render_env.step(actions)
                if track_trajectory and hasattr(render_env, "unwrapped"):
                    try:
                        # Try to get end-effector position from the environment
                        # This works for most ManiSkill manipulation tasks
                        env = render_env.unwrapped.base_env
                        if hasattr(env, "agent") and hasattr(env.agent, "tcp"):
                            # Get TCP (Tool Center Point) position - this is the end-effector
                            ee_pos = env.agent.tcp.pose.p
                            trajectories.append(
                                {
                                    "step": i,
                                    "ee_pos": ee_pos.cpu().numpy(),
                                    "actions": actions.cpu().numpy(),
                                    "rewards": rewards.cpu().numpy(),
                                    "dones": dones.cpu().numpy(),
                                }
                            )
                    except Exception as e:
                        pass
            elif env_type == "humanoid_bench":
                next_obs, rewards, dones, infos = render_env.step(actions)
                truncations = infos["time_outs"]
            else:
                next_obs, rewards, dones, _, infos = render_env.step(actions)

            if env_type == "mjx":
                try:
                    import jax.numpy as jnp

                    render_env.state.info["command"] = jnp.array([[1.0, 0.0, 0.0]])
                except (ImportError, AttributeError):
                    pass

            if env_type == "humanoid_bench":
                if i % 2 == 0:  # Record every 2nd frame to reduce video size
                    episode_frames.append(render_env.render())
            elif env_type == "maniskill":
                # For ManiSkill, try to get rgb_array if available
                if i % 2 == 0:
                    frame = render_env.render()
                    frame = frame.cpu().numpy() if frame is not None else None
                    if frame is not None:
                        # only take first env's frame
                        episode_frames.append(frame[0])
            elif env_type == "mjx":
                if i % 2 == 0:
                    renders.append(render_env.state)

            if dones.any():
                break
            obs = next_obs

        if env_type == "mjx":
            try:
                episode_frames = render_env.render_trajectory(renders)
            except AttributeError:
                episode_frames = []
        elif env_type == "humanoid_bench":
            # episode_frames is already populated
            pass
        elif env_type == "maniskill":
            # For ManiSkill, if we have RecordEpisode wrapper,
            # the video is automatically saved, but we return frames for wandb

            pass

        train_state.normalizer.train()

        if track_trajectory:
            return episode_frames, trajectories
        return episode_frames

    return render_with_rollout


def configure_platform(cfg: DictConfig) -> DictConfig:
    cfg.platform.amp_enabled = cfg.platform.amp_enabled and cfg.platform.cuda and torch.cuda.is_available()
    cfg.platform.amp_device = (
        "cuda"
        if cfg.platform.cuda and torch.cuda.is_available()
        else "mps" if cfg.platform.cuda and torch.backends.mps.is_available() else "cpu"
    )
    return cfg


def save_checkpoint(
    cfg: DictConfig,
    train_state: TrainState,
    global_step: int,
    run_name: str,
) -> None:
    if cfg.checkpoint_dir is None:
        return

    checkpoint_dir = Path(cfg.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = checkpoint_dir / f"{run_name}_step_{global_step}.pt"
    latest_path = checkpoint_dir / f"{run_name}_latest.pt"

    checkpoint = {
        "global_step": global_step,
        "actor_state_dict": train_state.actor.state_dict(),
        "old_actor_state_dict": train_state.old_actor.state_dict(),
        "critic_state_dict": train_state.critic.state_dict(),
        "critic_target_state_dict": train_state.critic_target.state_dict(),
        "actor_optimizer_state_dict": train_state.actor_optimizer.state_dict(),
        "critic_optimizer_state_dict": train_state.critic_optimizer.state_dict(),
        "normalizer_state_dict": (
            train_state.normalizer.state_dict() if hasattr(train_state.normalizer, "state_dict") else None
        ),
        "critic_normalizer_state_dict": (
            train_state.critic_normalizer.state_dict() if hasattr(train_state.critic_normalizer, "state_dict") else None
        ),
        "scaler_state_dict": train_state.scaler.state_dict(),
        "config": OmegaConf.to_container(cfg),
    }

    torch.save(checkpoint, checkpoint_path)
    print(f"Checkpoint saved to {checkpoint_path}")

    try:
        shutil.copyfile(checkpoint_path, latest_path)
        print(f"Latest checkpoint updated at {latest_path}")
    except Exception as e:
        print(f"Warning: Failed to update latest checkpoint symlink/copy: {e}")

    if cfg.get("wandb_upload_checkpoints", False):
        try:
            artifact_name = f"{run_name}_checkpoint"
            artifact = wandb.Artifact(
                name=artifact_name,
                type="model",
                description=f"Model checkpoint at step {global_step}",
                metadata={
                    "global_step": global_step,
                    "run_name": run_name,
                    "env_name": cfg.env.name,
                    "algorithm": "amdp_reppo",
                },
            )
            artifact.add_file(str(checkpoint_path), name=f"checkpoint_step_{global_step}.pt")
            wandb.log_artifact(artifact)
            print(f"Checkpoint uploaded to wandb as artifact: {artifact_name}")

        except Exception as e:
            print(f"Warning: Failed to upload checkpoint to wandb: {e}")
            # Don't fail the training if wandb upload fails


def load_checkpoint(
    cfg: DictConfig,
    train_state: TrainState,
    checkpoint_path: str,
    device: torch.device,
) -> tuple[TrainState, int]:
    """Load model checkpoint from disk."""
    print(f"Loading checkpoint from {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    train_state.actor.load_state_dict(checkpoint["actor_state_dict"])
    train_state.old_actor.load_state_dict(checkpoint["old_actor_state_dict"])
    train_state.critic.load_state_dict(checkpoint["critic_state_dict"])

    if "critic_target_state_dict" in checkpoint:
        train_state.critic_target.load_state_dict(checkpoint["critic_target_state_dict"])
    else:
        train_state.critic_target.load_state_dict(checkpoint["critic_state_dict"])

    train_state.actor_optimizer.load_state_dict(checkpoint["actor_optimizer_state_dict"])
    train_state.critic_optimizer.load_state_dict(checkpoint["critic_optimizer_state_dict"])

    if checkpoint.get("normalizer_state_dict") is not None and hasattr(train_state.normalizer, "load_state_dict"):
        train_state.normalizer.load_state_dict(checkpoint["normalizer_state_dict"])

    if checkpoint.get("critic_normalizer_state_dict") is not None and hasattr(
        train_state.critic_normalizer, "load_state_dict"
    ):
        train_state.critic_normalizer.load_state_dict(checkpoint["critic_normalizer_state_dict"])

    if checkpoint.get("scaler_state_dict") is not None:
        train_state.scaler.load_state_dict(checkpoint["scaler_state_dict"])

    global_step = checkpoint.get("global_step", 0)

    print(f"Checkpoint loaded successfully. Resuming from step {global_step}")

    return train_state, global_step


@hydra.main(
    version_base=None,
    config_path="../../config",
    config_name="amdp_reppo_torch",
)
def main(cfg):
    cfg = configure_platform(cfg)
    # Apply environment-specific hyperparameter overrides (e.g., vmin/vmax for different tasks)
    cfg = apply_env_specific_hyperparams(cfg)
    run_name = f"{cfg.name}_torch_{cfg.env.name}"

    if not hasattr(cfg, "checkpoint_dir"):
        cfg.checkpoint_dir = None
    if not hasattr(cfg, "checkpoint_path"):
        cfg.checkpoint_path = None
    if not hasattr(cfg, "save_checkpoint_interval"):
        cfg.save_checkpoint_interval = 0  # 0 means no periodic saving
    if not hasattr(cfg, "save_final_checkpoint"):
        cfg.save_final_checkpoint = True
    if not hasattr(cfg, "wandb_upload_checkpoints"):
        cfg.wandb_upload_checkpoints = False

    # If wandb_upload_checkpoints or save_final_checkpoint is enabled but no checkpoint_dir is set, use default
    if (cfg.wandb_upload_checkpoints or cfg.save_final_checkpoint) and cfg.checkpoint_dir is None:
        cfg.checkpoint_dir = "./checkpoints"

    if not hasattr(cfg, "eval_only"):
        cfg.eval_only = False
    if not hasattr(cfg, "eval_episodes"):
        cfg.eval_episodes = 10  # Number of episodes to evaluate when eval_only=True

    if not hasattr(cfg, "save_trajectories"):
        cfg.save_trajectories = False
    if not hasattr(cfg, "trajectory_dir"):
        cfg.trajectory_dir = "./trajectories"

    if not hasattr(cfg.hyperparameters, "render_interval"):
        cfg.hyperparameters.render_interval = 0  # 0 means no video recording
    if not hasattr(cfg.hyperparameters, "render_fps"):
        cfg.hyperparameters.render_fps = 30
    if not hasattr(cfg, "render_dir"):
        cfg.render_dir = None

    if cfg.eval_only and not cfg.checkpoint_path:
        raise ValueError("eval_only=True requires checkpoint_path to be specified")

    scaler = GradScaler(enabled=cfg.platform.amp_enabled and cfg.platform.amp_dtype == torch.float16)

    num_batches = cfg.hyperparameters.num_mini_batches
    batch_size = cfg.hyperparameters.num_envs * cfg.hyperparameters.num_steps // num_batches

    epochs_critic = cfg.hyperparameters.get("num_epochs_critic", cfg.hyperparameters.num_epochs)
    epochs_actor = cfg.hyperparameters.get("num_epochs_actor", cfg.hyperparameters.num_epochs)

    wandb.init(
        mode=cfg.wandb.mode,
        project=cfg.wandb.project,
        entity=cfg.wandb.entity,
        group=cfg.wandb.group,
        tags=[
            cfg.name,
            cfg.env.name,
            cfg.env.type,
            *cfg.tags,
        ],
        config=OmegaConf.to_container(cfg),
        name=f"{cfg.name}-torch-{cfg.env.name.lower()}",
        save_code=True,
    )

    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.backends.cudnn.deterministic = cfg.platform.torch_deterministic

    if not cfg.platform.cuda:
        device = torch.device("cpu")
    else:
        if torch.cuda.is_available():
            device = torch.device(f"cuda:{cfg.platform.device_rank}")
        elif torch.backends.mps.is_available():
            device = torch.device(f"mps:{cfg.platform.device_rank}")
        else:
            raise ValueError("No GPU available")
    print(f"Using device: {device}")

    envs, eval_envs, render_env = make_envs(cfg=cfg, device=device, seed=cfg.seed)

    use_staggered_resets = getattr(cfg.hyperparameters, "use_staggered_resets", False)
    staggered_mode = getattr(cfg.hyperparameters, "staggered_resets", "uniform_2")

    if use_staggered_resets:
        print(f"Applying Staggered Resets (Mode: {staggered_mode})")

        # 1. Determine Max Episode Steps
        # Try to find max_episode_steps from the env or config
        if hasattr(envs, "max_episode_steps"):
            max_episode_steps = envs.max_episode_steps
        elif hasattr(cfg.env, "max_episode_steps"):
            max_episode_steps = cfg.env.max_episode_steps
        else:
            # Fallback for ManiSkill/Gym wrappers if attribute is buried
            import mani_skill.utils.gym_utils as gym_utils

            try:
                max_episode_steps = gym_utils.find_max_episode_steps_value(envs)
            except:
                max_episode_steps = 1000  # Default fallback
                print(f"Warning: Could not determine max_episode_steps, defaulting to {max_episode_steps}")

        # 2. Determine Number of Stagger Blocks
        num_stagger_blocks = getattr(cfg.hyperparameters, "num_stagger_blocks", None)
        num_steps = cfg.hyperparameters.num_steps

        if num_stagger_blocks is not None and num_stagger_blocks > 0:
            num_groups = num_stagger_blocks
            print(f"Using specified num_stagger_blocks: {num_groups}")
        else:
            # Default: max_episode_steps // rollout_length
            calculated_default = max_episode_steps // num_steps if num_steps > 0 else 1
            num_groups = max(1, calculated_default)
            print(f"Defaulting stagger blocks to {num_groups} (H={max_episode_steps} / K={num_steps})")

        # 3. Calculate Offsets
        offsets = create_staggered_resets(
            max_episode_steps=max_episode_steps, num_envs=envs.num_envs, num_groups_requested=num_groups, device=device
        )

        # 4. Apply to Environment
        try:
            if hasattr(envs, "base_env"):
                envs.base_env._elapsed_steps = offsets
            elif hasattr(envs, "unwrapped"):
                envs.unwrapped._elapsed_steps = offsets
            else:
                envs._elapsed_steps = offsets
            print("Successfully applied staggered reset offsets.")

            # --- SANITY CHECK VERIFICATION ---
            # Verify that offsets were actually applied
            if hasattr(envs, "base_env"):
                steps_verify = envs.base_env._elapsed_steps
            elif hasattr(envs, "unwrapped"):
                steps_verify = envs.unwrapped._elapsed_steps
            else:
                steps_verify = envs._elapsed_steps

            print("\n" + "=" * 30)
            print("STAGGERED RESETS VERIFICATION")
            print("=" * 30)
            print(f"First 20 env elapsed steps: {steps_verify[:20].cpu().numpy()}")
            print(f"Mean elapsed step: {steps_verify.float().mean():.2f}")
            print(f"Max elapsed step: {steps_verify.max()}")
            print("=" * 30 + "\n")
            # ---------------------------------

        except AttributeError as e:
            print(f"Error applying staggered resets: Could not find '_elapsed_steps' in env. {e}")

    n_act = envs.num_actions
    n_obs = envs.num_obs if isinstance(envs.num_obs, int) else envs.num_obs[0]
    if envs.asymmetric_obs:
        n_critic_obs = (
            envs.num_privileged_obs if isinstance(envs.num_privileged_obs, int) else envs.num_privileged_obs[0]
        )
    else:
        n_critic_obs = n_obs

    if cfg.hyperparameters.normalize_env:
        obs_normalizer = EmpiricalNormalization(shape=n_obs, device=device)
        critic_obs_normalizer = EmpiricalNormalization(shape=n_critic_obs, device=device)
    else:
        obs_normalizer = nn.Identity()
        critic_obs_normalizer = nn.Identity()

    noise_schedule = hydra.utils.call(cfg.hyperparameters.diffusion.noise_schedule)

    sde_integrator = hydra.utils.get_method(cfg.hyperparameters.diffusion.sde_integrator)

    if cfg.hyperparameters.diffusion.learn_forward:
        forward_model: nn.Module = ControlNetwork(
            action_dim=n_act,
            observation_dim=n_obs,
            num_layers=cfg.hyperparameters.diffusion.score_model.num_layers,
            num_hid=cfg.hyperparameters.diffusion.score_model.num_hid,
            num_time_hid=cfg.hyperparameters.diffusion.score_model.num_time_hid,
            num_time_out=cfg.hyperparameters.diffusion.score_model.num_time_out,
            outer_clip=cfg.hyperparameters.diffusion.score_model.outer_clip,
            inner_clip=cfg.hyperparameters.diffusion.score_model.inner_clip,
            weight_init=cfg.hyperparameters.diffusion.score_model.weight_init,
            bias_init=cfg.hyperparameters.diffusion.score_model.bias_init,
            layer_norm=cfg.hyperparameters.diffusion.score_model.layer_norm,
            layer_norm_type=cfg.hyperparameters.diffusion.score_model.layer_norm_type,
            device=device,
        )
    else:
        forward_model = None

    if cfg.hyperparameters.diffusion.learn_backward:
        backward_model: nn.Module = ControlNetwork(
            action_dim=n_act,
            observation_dim=n_obs,
            num_layers=cfg.hyperparameters.diffusion.score_model.num_layers,
            num_hid=cfg.hyperparameters.diffusion.score_model.num_hid,
            num_time_hid=cfg.hyperparameters.diffusion.score_model.num_time_hid,
            num_time_out=cfg.hyperparameters.diffusion.score_model.num_time_out,
            outer_clip=cfg.hyperparameters.diffusion.score_model.outer_clip,
            inner_clip=cfg.hyperparameters.diffusion.score_model.inner_clip,
            weight_init=cfg.hyperparameters.diffusion.score_model.weight_init,
            bias_init=cfg.hyperparameters.diffusion.score_model.bias_init,
            layer_norm=cfg.hyperparameters.diffusion.score_model.layer_norm,
            layer_norm_type=cfg.hyperparameters.diffusion.score_model.layer_norm_type,
            device=device,
        )
    else:
        backward_model = None

    diffusion_model = DiffusionModel(
        action_dim=n_act,
        observation_dim=n_obs,
        fwd_model=forward_model,
        bwd_model=backward_model,
        diff_steps=cfg.hyperparameters.diffusion.diff_steps,
        noise_schedule=noise_schedule,
        device=device,
    )

    actor = DiffusionActor(
        action_dim=n_act,
        observation_dim=n_obs,
        diffusion_model=diffusion_model,
        sde_integrator=sde_integrator,
        ent_start=cfg.hyperparameters.ent_start,
        kl_start=cfg.hyperparameters.kl_start,
        device=device,
    )

    old_actor = copy.deepcopy(actor)
    qnet = Critic(
        n_obs=n_critic_obs,
        n_act=n_act,
        num_atoms=cfg.hyperparameters.num_bins,
        vmin=cfg.hyperparameters.vmin,
        vmax=cfg.hyperparameters.vmax,
        hidden_dim=cfg.hyperparameters.critic_hidden_dim,
        use_norm=cfg.hyperparameters.use_critic_norm,
        use_encoder_norm=False,
        encoder_layers=cfg.hyperparameters.num_critic_encoder_layers,
        head_layers=cfg.hyperparameters.num_critic_head_layers,
        pred_layers=cfg.hyperparameters.num_critic_pred_layers,
        device=device,
    )

    critic_target = copy.deepcopy(qnet)

    q_optimizer = optim.Adam(
        list(qnet.parameters()),
        lr=torch.tensor(cfg.hyperparameters.lr, device=device),
    )
    actor_optimizer = optim.Adam(
        list(actor.parameters()),
        lr=torch.tensor(cfg.hyperparameters.lr, device=device),
    )

    if envs.asymmetric_obs:
        (obs, *_), (critic_obs, *_) = envs.reset_with_critic_obs()
        # TODO:?
        # obs, critic_obs = envs.reset_with_critic_obs()
        # obs = torch.as_tensor(obs, device=device, dtype=torch.float)
        critic_obs = torch.as_tensor(critic_obs, device=device, dtype=torch.float)
    else:
        if cfg.env.type == "maniskill":
            (obs, *_) = envs.reset()
            critic_obs = obs
        else:
            obs = envs.reset()
            critic_obs = obs

    train_state = TrainState(
        obs=obs,
        critic_obs=critic_obs,
        actor=actor,
        old_actor=old_actor,
        critic=qnet,
        critic_target=critic_target,
        normalizer=obs_normalizer,
        critic_normalizer=critic_obs_normalizer,
        actor_optimizer=actor_optimizer,
        critic_optimizer=q_optimizer,
        device=device,
        scaler=scaler,
    )

    print(
        summary(
            train_state.critic,
            input_data=(critic_obs, torch.zeros((cfg.hyperparameters.num_envs, n_act), device=device)),
            depth=10,
        )
    )
    print(summary(train_state.actor, input_data=(obs,), depth=10))

    collect_fn = make_collect_fn(cfg, envs, cfg.env.type)
    postprocess_fn = make_postprocess_fn(cfg, envs)
    update_critic = make_critic_update_fn(cfg, train_state)
    update_actor = make_actor_update_fn(cfg, train_state)

    sde_evaluate = make_sde_eval_fn(cfg, eval_envs, cfg.env.type)
    best_of_n_evaluate = make_best_of_n_eval_fn(cfg, eval_envs, cfg.env.type)
    evaluate = make_evaluate_fn(cfg, eval_envs, cfg.env.type)
    render_rollout = make_render_fn(cfg, render_env, cfg.env.type)

    if cfg.platform.compile:
        mode = "max-autotune-no-cudagraphs"
        update_critic = torch.compile(update_critic, mode=mode)
        update_actor = torch.compile(update_actor, mode=mode)
        postprocess_fn = torch.compile(postprocess_fn, mode=mode)
        train_state.compile()

    if cfg.checkpoint_path:
        train_state, global_step = load_checkpoint(cfg, train_state, cfg.checkpoint_path, device)
    else:
        global_step = 0

    if cfg.eval_only:
        print(f"Running evaluation only mode with {cfg.eval_episodes} episodes...")

        eval_returns = []
        eval_lengths = []
        eval_successes = []
        all_trajectories = []

        track_traj = cfg.save_trajectories and cfg.env.type == "maniskill"

        if track_traj:
            print(f"Trajectory tracking enabled. Will save to {cfg.trajectory_dir}")

        for episode in range(cfg.eval_episodes):
            print(f"Running evaluation episode {episode + 1}/{cfg.eval_episodes}")

            eval_avg_return, eval_avg_length, eval_info = evaluate(train_state)

            eval_returns.append(eval_avg_return)
            eval_lengths.append(eval_avg_length)

            success_rate = eval_info.get("success", 0.0)
            if isinstance(success_rate, torch.Tensor):
                success_rate = success_rate.item()
            elif isinstance(success_rate, np.ndarray):
                success_rate = success_rate.mean()
            eval_successes.append(success_rate)

            print(
                f"Episode {episode + 1}: Return={eval_avg_return:.2f}, Length={eval_avg_length:.2f}, Success={success_rate:.2f}"
            )

            if (cfg.hyperparameters.render_interval > 0) or track_traj:
                if track_traj:
                    print(f"Recording video and tracking trajectory for episode {episode + 1}")
                else:
                    print(f"Recording video for episode {episode + 1}")
                try:
                    result = render_rollout(train_state, track_trajectory=track_traj)

                    if track_traj:
                        renders, trajectories = result
                        if trajectories:
                            all_trajectories.append(trajectories)
                            print(f"  Captured {len(trajectories)} trajectory steps")
                    else:
                        renders = result

                    if renders and len(renders) > 0:
                        if cfg.env.type == "humanoid_bench":
                            video_array = np.array(renders)
                            if video_array.ndim == 4:  # (T, H, W, C)
                                video_array = video_array.transpose(0, 3, 1, 2)
                            render_video = wandb.Video(
                                video_array,
                                fps=cfg.hyperparameters.render_fps,
                                format="gif",
                            )
                            wandb.log({f"eval_video_episode_{episode + 1}": render_video})
                            print(f"Video recorded with {len(renders)} frames")
                        else:
                            print("Video saved locally (if recording enabled)")
                except Exception as e:
                    print(f"Error recording video for episode {episode + 1}: {e}")

        mean_return = np.mean(eval_returns)
        std_return = np.std(eval_returns)
        mean_length = np.mean(eval_lengths)
        std_length = np.std(eval_lengths)
        mean_success = np.mean(eval_successes)
        std_success = np.std(eval_successes)

        print("\n" + "=" * 50)
        print("EVALUATION SUMMARY")
        print("=" * 50)
        print(f"Episodes evaluated: {cfg.eval_episodes}")
        print(f"Mean return: {mean_return:.2f} ± {std_return:.2f}")
        print(f"Mean length: {mean_length:.2f} ± {std_length:.2f}")
        print(f"Mean success: {mean_success:.2f} ± {std_success:.2f}")
        print("=" * 50)

        wandb.log(
            {
                "eval_summary/mean_return": mean_return,
                "eval_summary/std_return": std_return,
                "eval_summary/mean_length": mean_length,
                "eval_summary/std_length": std_length,
                "eval_summary/mean_success": mean_success,
                "eval_summary/std_success": std_success,
                "eval_summary/episodes": cfg.eval_episodes,
            }
        )

        if cfg.save_trajectories and all_trajectories:
            print("\n" + "=" * 50)
            print("SAVING TRAJECTORIES")
            print("=" * 50)

            episode_info = {
                "returns": eval_returns,
                "lengths": eval_lengths,
                "successes": eval_successes,
                "mean_return": mean_return,
                "std_return": std_return,
                "mean_success": mean_success,
            }

            try:
                plot_enabled = cfg.get("plot_trajectories", True)
                traj_file, plot_file = save_and_plot_trajectories(
                    all_trajectories=all_trajectories,
                    save_dir=cfg.trajectory_dir,
                    env_name=cfg.env.name,
                    episode_info=episode_info,
                    plot_trajectories=plot_enabled,
                )

                if plot_file:
                    try:
                        import matplotlib.pyplot as plt

                        trajectory_image = wandb.Image(str(plot_file))
                        wandb.log({"eval_trajectories/summary_plot": trajectory_image})
                        print(f"Uploaded trajectory plot to wandb")
                    except Exception as e:
                        print(f"Could not upload trajectory plot to wandb: {e}")

            except Exception as e:
                print(f"Error saving trajectories: {e}")
                import traceback

                traceback.print_exc()

            print("=" * 50)

        print("Evaluation completed. Exiting.")
        return

    total_env_steps = (
        cfg.hyperparameters.total_time_steps // (cfg.hyperparameters.num_envs * cfg.hyperparameters.num_steps) + 1
    )

    pbar = tqdm.tqdm(total=cfg.hyperparameters.total_time_steps, initial=global_step)
    start_time = None
    desc = ""

    eval_interval = total_env_steps // cfg.hyperparameters.num_eval

    def run_evaluation(global_step):
        logs = {}
        print(f"Evaluating at global step {global_step}")

        sde_metrics = sde_evaluate(train_state)

        bon_metrics = {}
        if hasattr(cfg.hyperparameters, "best_of_n_values") and cfg.hyperparameters.best_of_n_values:
            for best_of_n in cfg.hyperparameters.best_of_n_values:
                bon_result = best_of_n_evaluate(train_state, best_of_n=best_of_n)
                bon_suffix = f"bon_{int(best_of_n):03d}"
                bon_metrics.update({f"{k}_{bon_suffix}": v for k, v in bon_result.items()})

        eval_info = {**sde_metrics, **bon_metrics}
        eval_avg_return = sde_metrics["episode_return"]
        eval_avg_length = sde_metrics["episode_length"]

        if cfg.env.type == "humanoid_bench":
            # NOTE: Hacky way of evaluating performance, but just works
            if envs.asymmetric_obs:
                obs, critic_obs = envs.reset_with_critic_obs()
            else:
                obs = envs.reset()

        logs["eval/episode_return"] = eval_avg_return
        logs["eval/episode_length"] = eval_avg_length

        if "success" in sde_metrics:
            logs["eval/success"] = sde_metrics["success"]
        if "info_return" in sde_metrics:
            logs["eval/info_return"] = sde_metrics["info_return"]

        for key, value in bon_metrics.items():
            logs[f"eval/{key}"] = value

        print(
            f"Eval return: {eval_avg_return:.2f}, length: {eval_avg_length:.2f}, "
            f"env steps: {global_step * cfg.hyperparameters.num_envs * cfg.hyperparameters.num_steps}, "
            f"success rate: {sde_metrics.get('success', 0.0):.2f}"
        )
        return logs

    def run_rendering(global_step):
        logs = {}
        print(f"Recording video at global step {global_step}")
        try:
            renders = render_rollout(train_state)
            if renders and len(renders) > 0:
                # Convert renders to numpy array with correct format for wandb
                if cfg.env.type == "humanoid_bench":
                    # renders is a list of (H, W, C) arrays
                    video_array = np.array(renders)
                    # Convert to (T, C, H, W) format for wandb
                    if video_array.ndim == 4:  # (T, H, W, C)
                        video_array = video_array.transpose(0, 3, 1, 2)

                    render_video = wandb.Video(
                        video_array,
                        fps=cfg.hyperparameters.render_fps,
                        format="gif",
                    )
                    logs["render_video"] = render_video
                    print(f"Video recorded with {len(renders)} frames")

                elif cfg.env.type == "maniskill":
                    if isinstance(renders, list) and len(renders) > 0:
                        video_array = np.array(renders)
                        if video_array.ndim == 4:  # (T, H, W, C)
                            video_array = video_array.transpose(0, 3, 1, 2)
                        elif video_array.ndim == 3:  # Single frame (H, W, C)
                            video_array = video_array.transpose(2, 0, 1)[None, ...]

                        render_video = wandb.Video(
                            video_array,
                            fps=cfg.hyperparameters.render_fps,
                            format="gif",
                        )
                        logs["render_video"] = render_video
                        print(f"Video recorded with {len(renders)} frames")
                    else:
                        print("ManiSkill video saved locally (if RecordEpisode wrapper enabled)")

                elif cfg.env.type == "mjx":
                    if hasattr(renders, "shape") and len(renders) > 0:
                        video_array = np.array(renders)
                        if video_array.ndim == 4:
                            video_array = video_array.transpose(0, 3, 1, 2)
                        render_video = wandb.Video(
                            video_array,
                            fps=cfg.hyperparameters.render_fps,
                            format="gif",
                        )
                        logs["render_video"] = render_video
                        print(f"Video recorded with {len(renders)} frames")
            else:
                print("No frames rendered - skipping video logging")
                if cfg.env.type == "maniskill" and cfg.render_dir:
                    print(f"Check {cfg.render_dir} for locally saved videos")
        except Exception as e:
            print(f"Error recording video: {e}")
        return logs

    # Init or loaded checkpoint have finished some number of steps
    # check if we should eval before starting to train (i.e. step zero) to get baseline
    initial_logs = {}
    if eval_interval > 0 and global_step % eval_interval == 0:
        initial_logs.update(run_evaluation(global_step))
    if cfg.hyperparameters.render_interval > 0 and global_step % cfg.hyperparameters.render_interval == 0:
        initial_logs.update(run_rendering(global_step))

    if initial_logs:
        wandb.log(
            initial_logs,
            step=global_step * cfg.hyperparameters.num_envs * cfg.hyperparameters.num_steps,
        )

    while global_step < total_env_steps:
        if start_time is None and global_step >= cfg.measure_burnin:
            start_time = time.time()
            measure_burnin = global_step

        train_state, transition, infos = collect_fn(train_state)
        data = postprocess_fn(train_state, transition)

        # 1. Critic Update
        for _ in range(epochs_critic):
            indices = torch.randperm(data.shape[0], device=device)
            for j in range(num_batches):
                mini_batch = data[indices[j * batch_size : (j + 1) * batch_size]]
                critic_logs_dict = update_critic(mini_batch)

        # 2. Target Critic Polyak Update
        with torch.no_grad():
            polyak = cfg.hyperparameters.polyak
            for param, target_param in zip(train_state.critic.parameters(), train_state.critic_target.parameters()):
                target_param.data.mul_(1 - polyak).add_(param.data, alpha=polyak)

        # 3. Update Buffer with Target Q Gradients
        data = update_Q(data, train_state, cfg)

        # 4. Actor Update
        for _ in range(epochs_actor):
            indices = torch.randperm(data.shape[0], device=device)
            for j in range(num_batches):
                mini_batch = data[indices[j * batch_size : (j + 1) * batch_size]]
                actor_logs_dict = update_actor(mini_batch)

        logs_dict = {**critic_logs_dict, **actor_logs_dict}

        for param, target_param in zip(actor.parameters(), old_actor.parameters()):
            target_param.data.copy_(param.data)
            target_param.requires_grad = False

        # we have finished one update, so increment the step counter to synchronize the eval to the
        # number of taken transitions
        global_step += 1
        env_steps = global_step * cfg.hyperparameters.num_envs * cfg.hyperparameters.num_steps

        logs = {}
        if start_time is not None:
            speed = (
                cfg.hyperparameters.num_envs
                * cfg.hyperparameters.num_steps
                * (global_step - measure_burnin)
                / (time.time() - start_time)
            )
            pbar.set_description(f"{speed: 4.4f} sps, " + desc)
            logs["speed"] = speed
        with torch.inference_mode():
            logs.update(
                {
                    "critic/qf_loss": logs_dict["qf_loss"].mean(),
                    "critic/qf_max": logs_dict["qf_max"].max(),
                    "critic/qf_min": logs_dict["qf_min"].min(),
                    "critic/qf_mean": logs_dict["qf_mean"].mean(),
                    "critic/embedding_loss": logs_dict["embedding_loss"].mean(),
                    "critic/critic_grad_norm": logs_dict["critic_grad_norm"].mean(),
                    "actor/actor_loss": logs_dict.get("actor_loss", torch.tensor(0.0)).mean(),
                    "actor/adjoint_loss": logs_dict.get("adjoint_loss", torch.tensor(0.0)).mean(),
                    "actor/kl_loss": logs_dict.get("kl_loss", torch.tensor(0.0)).mean(),
                    "actor/actor_grad_norm": logs_dict.get("actor_grad_norm", torch.tensor(0.0)).mean(),
                    "actor/entropy": logs_dict.get("entropy", torch.tensor(0.0)).mean(),
                    "actor/optimal_entropy": logs_dict.get("optimal_entropy", torch.tensor(0.0)).mean(),
                    "actor/temperature": logs_dict.get("temperature", torch.tensor(0.0)).mean(),
                    "actor/lagrangian": logs_dict.get("lagrangian", torch.tensor(0.0)).mean(),
                    "actor/dual_lm": logs_dict.get("dual_lm", torch.tensor(0.0)).mean(),
                    "actor/opt_lm": logs_dict.get("opt_lm", torch.tensor(0.0)).mean(),
                    "train/rewards_batch": data["rewards"].mean(),
                }
            )

            if cfg.env.type == "maniskill":
                logs.update(
                    {
                        "train/return": torch.stack([info["log_info"]["return"] for info in infos]).mean(),
                        "train/episode_len": torch.stack([info["log_info"]["episode_len"] for info in infos])
                        .float()
                        .mean(),
                        "train/success": torch.stack([info["log_info"]["success"] for info in infos]).float().mean(),
                    }
                )

            if eval_interval > 0 and global_step % eval_interval == 0:
                logs.update(run_evaluation(global_step))

            if cfg.hyperparameters.render_interval > 0 and global_step % cfg.hyperparameters.render_interval == 0:
                logs.update(run_rendering(global_step))

            if cfg.save_checkpoint_interval > 0 and global_step % (eval_interval * cfg.save_checkpoint_interval) == 0:
                save_checkpoint(cfg, train_state, global_step, run_name)

        wandb.log(
            {
                "frame": env_steps,
                **logs,
            },
            step=env_steps,
        )

        pbar.update(n=cfg.hyperparameters.num_envs * cfg.hyperparameters.num_steps)

    if cfg.save_final_checkpoint:
        save_checkpoint(cfg, train_state, global_step, run_name)
        print("Training completed. Final checkpoint saved.")

    wandb.finish()


if __name__ == "__main__":
    main()
