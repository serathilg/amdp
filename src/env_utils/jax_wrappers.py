from functools import partial
from typing import Any, Optional, Tuple, Union

import chex
import gymnasium as gym
import gymnax
import jax
import jax.numpy as jnp
import mani_skill.envs
import numpy as np
import torch
from flax import struct
from gymnasium.wrappers import TimeLimit
from gymnax.environments import environment, spaces
from gymnax.environments.environment import Environment
from gymnax.environments.spaces import Box
from jax.dlpack import from_dlpack
from jaxtyping import Array, Bool, Float, PRNGKeyArray

# The wrapper reset/step signatures below share one contract (annotations are
# documentation, not runtime-checked):
#   reset(key) -> (obs, critic_obs, state)
#   step(key, state, action) -> (obs, critic_obs, state, reward, done, info)
# with obs/critic_obs "envs obs"/"envs critic_obs", reward "envs", done "envs" bool.
ObsBatch = Float[Array, "envs obs"]
CriticObsBatch = Float[Array, "envs critic_obs"]
ActionBatch = Float[Array, "envs act"]
ResetReturn = Tuple[ObsBatch, CriticObsBatch, Any]
StepReturn = Tuple[
    ObsBatch, CriticObsBatch, Any, Float[Array, "envs"], Bool[Array, "envs"], dict
]

from mani_skill.utils import gym_utils
from mani_skill.utils.wrappers.flatten import FlattenActionSpaceWrapper
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv
from ml_collections import ConfigDict
from mujoco_playground import MjxEnv, registry
from mujoco_playground._src.wrapper import Wrapper, wrap_for_brax_training
from stable_baselines3.common.vec_env import SubprocVecEnv
from torch.utils.dlpack import to_dlpack

from src.env_utils.maniskill_utils import maniskill_env_kwargs
from src.env_utils.mixed_vec_env import MixedVecEnv
from src.env_utils.mjx_patches import patch_touch_sensor_matmuls


class MjxGymnaxWrapper(Environment):
    def __init__(
        self,
        env_or_name: str | MjxEnv,
        episode_length: int = 1000,
        action_repeat: int = 1,
        push_distractions: bool = False,
        config: dict = None,
        asymmetric_observation: bool = False,
    ):
        # MJX as shipped cannot be vmapped over a seed axis on top of its env axis.
        patch_touch_sensor_matmuls()
        if isinstance(env_or_name, str):
            if config is None:
                config = registry.get_default_config(env_or_name)
                is_humanoid_task = env_or_name in [
                    "G1JoystickRoughTerrain",
                    "G1JoystickFlatTerrain",
                    "T1JoystickRoughTerrain",
                    "T1JoystickFlatTerrain",
                ]
                if is_humanoid_task:
                    config.push_config.enable = push_distractions
            else:
                config = ConfigDict(config)
            env = registry.load(env_or_name, config=config)
            if episode_length is not None:
                env = wrap_for_brax_training(
                    env, episode_length=episode_length, action_repeat=action_repeat
                )
            self.env = env
        else:
            self.env = env_or_name
        if isinstance(self.env.observation_size, int):
            self.dict_obs = False
        else:
            self.dict_obs = True
        if asymmetric_observation:
            self.dict_obs_key = "privileged_state"
        else:
            self.dict_obs_key = "state"
        print(self.dict_obs_key)
        super().__init__()

    def action_space(self, params):
        return gymnax.environments.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self.env.action_size,),
        )

    def observation_space(self, params):
        if self.dict_obs:
            return Box(
                low=-float("inf"),
                high=float("inf"),
                shape=self.env.observation_size["state"],
            ), Box(
                low=-float("inf"),
                high=float("inf"),
                shape=self.env.observation_size[self.dict_obs_key],
            )
        else:
            return Box(
                low=-float("inf"),
                high=float("inf"),
                shape=(self.env.observation_size,),
            ), Box(
                low=-float("inf"),
                high=float("inf"),
                shape=(self.env.observation_size,),
            )

    @property
    def default_params(self) -> gymnax.EnvParams:
        return gymnax.EnvParams()

    def reset(self, key: PRNGKeyArray) -> ResetReturn:
        state = self.env.reset(key)
        obs = state.obs if not self.dict_obs else state.obs["state"]
        critic_obs = state.obs if not self.dict_obs else state.obs[self.dict_obs_key]
        return obs, critic_obs, state

    def step(self, key: PRNGKeyArray, state, action: ActionBatch) -> StepReturn:
        state = self.env.step(state, action)
        obs = state.obs if not self.dict_obs else state.obs["state"]
        critic_obs = state.obs if not self.dict_obs else state.obs[self.dict_obs_key]
        return (
            obs,
            critic_obs,
            state,
            state.reward,
            state.done > 0.5,
            {},
        )


@struct.dataclass
class LogEnvState:
    env_state: environment.EnvState
    episode_returns: jnp.ndarray
    episode_lengths: jnp.ndarray
    episode_successes: jnp.ndarray
    returned_episode_returns: jnp.ndarray
    returned_episode_lengths: jnp.ndarray
    returned_episode_successes: jnp.ndarray
    timestep: jnp.ndarray
    truncated: jnp.ndarray
    info: Any = None

    def unwrapped(self):
        return self.env_state

    def set_env_state(self, env_state):
        if hasattr(self.env_state, "set_env_state"):
            return self.replace(env_state=self.env_state.set_env_state(env_state))
        return self.replace(env_state=env_state)


def init_episode_stats(num_envs: int, env_state: environment.EnvState) -> LogEnvState:
    """Build the initial episode-statistics state for `num_envs` fresh episodes."""
    return LogEnvState(
        env_state=env_state,
        episode_returns=jnp.zeros((num_envs,)),
        episode_lengths=jnp.zeros((num_envs,), dtype=jnp.int32),
        episode_successes=jnp.zeros((num_envs,), dtype=jnp.bool_),
        returned_episode_returns=jnp.zeros((num_envs,)),
        returned_episode_lengths=jnp.zeros((num_envs,), dtype=jnp.int32),
        returned_episode_successes=jnp.zeros((num_envs,), dtype=jnp.bool_),
        timestep=jnp.zeros((num_envs,), dtype=jnp.int32),
        truncated=jnp.ones((num_envs,), dtype=jnp.float32),
        info={
            "returned_episode": jnp.zeros((num_envs,), dtype=jnp.bool_),
            "returned_episode_returns": jnp.zeros((num_envs,)),
            "returned_episode_successes": jnp.zeros((num_envs,), dtype=jnp.bool_),
            "timestep": jnp.zeros((num_envs,), dtype=jnp.int32),
            "returned_episode_lengths": jnp.zeros((num_envs,), dtype=jnp.int32),
        },
    )


def update_episode_stats(
    state: LogEnvState,
    env_state: environment.EnvState,
    reward: chex.Array,
    done: chex.Array,
    info: dict,
) -> Tuple[LogEnvState, dict]:
    """Carry the running episode return/length/success across one step.

    When an episode ends, the running values are latched into the `returned_*` fields
    (which the trainer averages for its train/episode_* metrics) and the running ones
    reset to zero. `info` is updated in place and returned alongside the new state.
    """
    # An episode ends on `done` OR `truncation`. They coincide for JAX-native envs
    # (truncation is a subset of done there), but host-stepped ManiSkill reports the
    # RL-done as 0 under partial_reset (every end bootstraps like a truncation), so
    # latching on `done` alone would never fire there.
    done = jnp.logical_or(
        jnp.asarray(done, dtype=bool),
        jnp.asarray(env_state.info["truncation"], dtype=bool),
    )
    new_episode_return = state.episode_returns + reward
    new_episode_length = state.episode_lengths + 1
    new_episode_success = state.episode_successes | env_state.info.get(
        "success", jnp.zeros_like(state.episode_successes)
    )
    info["returned_episode_returns"] = (
        state.returned_episode_returns * (1 - done) + new_episode_return * done
    )
    info["returned_episode_lengths"] = (
        state.returned_episode_lengths * (1 - done) + new_episode_length * done
    )
    info["returned_episode_successes"] = (
        state.returned_episode_successes & (~done) | new_episode_success & done
    )
    info["timestep"] = state.timestep
    info["returned_episode"] = done
    state = LogEnvState(
        env_state=env_state,
        episode_returns=new_episode_return * (1 - done),
        episode_lengths=new_episode_length * (1 - done),
        episode_successes=new_episode_success & (~done),
        returned_episode_returns=state.returned_episode_returns * (1 - done)
        + new_episode_return * done,
        returned_episode_lengths=state.returned_episode_lengths * (1 - done)
        + new_episode_length * done,
        returned_episode_successes=state.returned_episode_successes & (~done)
        | new_episode_success & done,
        timestep=state.timestep + 1,
        truncated=env_state.info["truncation"],
        info=info,
    )
    return state, info


class LogWrapper(Wrapper):
    """Log the episode returns and lengths."""

    def __init__(self, env: environment.Environment, num_envs: int):
        super().__init__(env)
        self.num_envs = num_envs

    # Not jitted: for a host-stepped env this chain runs eagerly around a plain-Python
    # env.step. In the JAX-native path it is traced by the caller's jit either way.
    def reset(self, key: PRNGKeyArray) -> ResetReturn:
        obs, critic_obs, env_state = self.env.reset(key)
        return obs, critic_obs, init_episode_stats(self.num_envs, env_state)

    def step(
        self,
        key: PRNGKeyArray,
        state: environment.EnvState,
        action: ActionBatch,
    ) -> StepReturn:
        obs, critic_obs, env_state, reward, done, info = self.env.step(
            key, state.env_state, action
        )
        state, info = update_episode_stats(state, env_state, reward, done, info)
        return obs, critic_obs, state, reward, done, info


class ClipAction(Wrapper):
    def __init__(self, env, low=-0.999, high=0.999):
        super().__init__(env)
        self.low = low
        self.high = high

    def step(self, key: PRNGKeyArray, state, action: ActionBatch) -> StepReturn:
        # Clips to the fixed (low, high) range rather than the env's action-space
        # bounds: every env here uses [-1, 1] actions, and the 0.999 margin keeps the
        # squashed actions strictly inside the open interval.
        action = jnp.clip(action, self.low, self.high)
        return self.env.step(key, state, action)


@struct.dataclass
class NormalizeVecObsEnvState:
    mean: jnp.ndarray
    var: jnp.ndarray
    critic_mean: jnp.ndarray
    critic_var: jnp.ndarray
    count: float
    env_state: environment.EnvState
    truncated: float
    info: Any = None

    def unwrapped(self):
        return self.env_state.unwrapped()

    def set_env_state(self, env_state):
        return self.replace(env_state=self.env_state.set_env_state(env_state))


def welford_update(
    mean: Float[Array, "obs"],
    var: Float[Array, "obs"],
    count: Float[Array, ""] | float,
    obs: Float[Array, "envs obs"],
) -> tuple[Float[Array, "obs"], Float[Array, "obs"]]:
    """Pool the running (mean, var, count) with a batch of observations."""
    batch_mean = jnp.mean(obs, axis=0)
    batch_var = jnp.var(obs, axis=0)
    batch_count = obs.shape[0]

    delta = batch_mean - mean
    tot_count = count + batch_count

    new_mean = mean + delta * batch_count / tot_count
    m_a = var * count
    m_b = batch_var * batch_count
    M2 = m_a + m_b + jnp.square(delta) * count * batch_count / tot_count
    new_var = M2 / tot_count

    return new_mean, new_var


def normalize_obs(
    obs: Float[Array, "envs obs"],
    mean: Float[Array, "obs"],
    var: Float[Array, "obs"],
    eps: float = 1e-2,
) -> Float[Array, "envs obs"]:
    """Standardize observations with the running statistics."""
    return (obs - mean) / jnp.sqrt(var + eps)


def init_norm_stats(
    obs: Float[Array, "envs obs"],
    critic_obs: Float[Array, "envs critic_obs"],
    env_state: environment.EnvState,
    params: NormalizeVecObsEnvState | None = None,
) -> NormalizeVecObsEnvState:
    """Build the initial normalizer state, seeding the statistics from `params` if given
    (e.g. an evaluation reset reusing the training normalizer) or from the reset batch."""
    if params is not None:
        mean, var = params.mean, params.var
        critic_mean, critic_var = params.critic_mean, params.critic_var
        count = params.count
    else:
        mean = jnp.mean(obs, axis=0)
        var = jnp.var(obs, axis=0)
        critic_mean = jnp.mean(critic_obs, axis=0)
        critic_var = jnp.var(critic_obs, axis=0)
        count = obs.shape[0]
    return NormalizeVecObsEnvState(
        mean=mean,
        var=var,
        critic_mean=critic_mean,
        critic_var=critic_var,
        count=count,
        env_state=env_state,
        truncated=env_state.truncated,
        info=env_state.info,
    )


def update_norm_stats(
    state: NormalizeVecObsEnvState,
    obs: Float[Array, "envs obs"],
    critic_obs: Float[Array, "envs critic_obs"],
    env_state: environment.EnvState,
    update_stats: bool = True,
) -> NormalizeVecObsEnvState:
    """Advance the running normalizer statistics by one step's observation batch."""
    if update_stats:
        new_mean, new_var = welford_update(state.mean, state.var, state.count, obs)
        new_critic_mean, new_critic_var = welford_update(
            state.critic_mean, state.critic_var, state.count, critic_obs
        )
        new_count = state.count + obs.shape[0]
    else:
        new_mean, new_var = state.mean, state.var
        new_critic_mean, new_critic_var = state.critic_mean, state.critic_var
        new_count = state.count
    return NormalizeVecObsEnvState(
        mean=new_mean,
        var=new_var,
        critic_mean=new_critic_mean,
        critic_var=new_critic_var,
        count=new_count,
        env_state=env_state,
        truncated=env_state.truncated,
        info=env_state.info,
    )


def wrap_reset_stats(
    obs: Float[Array, "envs obs"],
    critic_obs: Float[Array, "envs critic_obs"],
    raw_state: environment.EnvState,
    num_envs: int,
    norm_params: NormalizeVecObsEnvState | None = None,
    normalize: bool = True,
) -> tuple[
    Float[Array, "envs obs"], Float[Array, "envs critic_obs"], NormalizeVecObsEnvState
]:
    """Compose the LogWrapper+NormalizeVec reset math for raw reset results.

    A host-stepped environment resets outside the traced graph, so the wrapper classes
    never see it; this applies the identical statistics math (per seed, under `vmap`)
    to the raw outputs. Mirrors `LogWrapper.reset` followed by `NormalizeVec.reset`.
    """
    log_state = init_episode_stats(num_envs, raw_state)
    state = init_norm_stats(obs, critic_obs, log_state, norm_params)
    if normalize:
        obs = normalize_obs(obs, state.mean, state.var)
        critic_obs = normalize_obs(critic_obs, state.critic_mean, state.critic_var)
    return obs, critic_obs, state


def wrap_step_stats(
    state: NormalizeVecObsEnvState,
    obs: Float[Array, "envs obs"],
    critic_obs: Float[Array, "envs critic_obs"],
    raw_state: environment.EnvState,
    reward: Float[Array, "envs"],
    done: Float[Array, "envs"],
    info: dict,
    update_stats: bool = True,
    normalize: bool = True,
) -> tuple[
    Float[Array, "envs obs"],
    Float[Array, "envs critic_obs"],
    NormalizeVecObsEnvState,
    Float[Array, "envs"],
    Float[Array, "envs"],
    dict,
]:
    """Compose the LogWrapper+NormalizeVec step math for raw step results.

    Host-stepped counterpart of `LogWrapper.step` followed by `NormalizeVec.step`
    (action clipping happens before the raw step and is the caller's concern).
    """
    log_state, info = update_episode_stats(state.env_state, raw_state, reward, done, info)
    state = update_norm_stats(state, obs, critic_obs, log_state, update_stats)
    if normalize:
        obs = normalize_obs(obs, state.mean, state.var)
        critic_obs = normalize_obs(critic_obs, state.critic_mean, state.critic_var)
    return obs, critic_obs, state, reward, done, info


class NormalizeVec(Wrapper):
    def __init__(self, env, enable=True):
        super().__init__(env)
        self.enable = enable

    def reset(self, key: PRNGKeyArray, params=None) -> ResetReturn:
        obs, critic_obs, env_state = self.env.reset(key)
        state = init_norm_stats(obs, critic_obs, env_state, params)
        return (
            (
                normalize_obs(obs, state.mean, state.var),
                normalize_obs(critic_obs, state.critic_mean, state.critic_var),
                state,
            )
            if self.enable
            else (obs, critic_obs, state)
        )

    def step(
        self, key: PRNGKeyArray, state, action: ActionBatch, update_stats: bool = True
    ) -> StepReturn:
        obs, critic_obs, env_state, reward, done, info = self.env.step(key, state.env_state, action)
        state = update_norm_stats(state, obs, critic_obs, env_state, update_stats)
        return (
            (
                normalize_obs(obs, state.mean, state.var),
                normalize_obs(critic_obs, state.critic_mean, state.critic_var),
                state,
                reward,
                done,
                info,
            )
            if self.enable
            else (obs, critic_obs, state, reward, done, info)
        )


def to_jax(tensor):
    if isinstance(tensor, dict):
        return {k: to_jax(v) for k, v in tensor.items()}
    if not isinstance(tensor, torch.Tensor):
        return tensor
    # Ensure tensor is on the correct device and contiguous
    return from_dlpack(to_dlpack(tensor.contiguous()))


def to_torch(array):
    if isinstance(array, dict):
        return {k: to_torch(v) for k, v in array.items()}
    if not isinstance(array, (jax.Array, jnp.ndarray)):
        return array
    return torch.from_dlpack(jax.dlpack.to_dlpack(array))


@struct.dataclass
class ManiSkillEnvState:
    obs: Union[jnp.ndarray, dict]
    info: dict
    done: jnp.ndarray
    truncated: jnp.ndarray
    _wrapper: Any = struct.field(pytree_node=False)  # Reference to sync with engine

    def unwrapped(self):
        return self

    def set_env_state(self, new_state):
        """
        Sync manually set info['steps'] back to ManiSkill's internal PyTorch counters.
        """
        if "steps" in new_state.info:
            self._wrapper._sync_steps_to_torch(new_state.info["steps"])
        return new_state


class ManiSkillGymnaxWrapper:
    def __init__(
        self,
        env_name: str,
        num_envs: int,
        reconfiguration_freq: Optional[int] = None,
        partial_reset: bool = True,
        asymmetric_observation: bool = False,
        env_kwargs: dict = None,
        has_final_obs: bool = False,
        ignore_terminations: bool = True,
    ):
        self.env_name = env_name
        self.num_envs = num_envs
        self.partial_reset = partial_reset
        self.asymmetric_observation = asymmetric_observation
        self.has_final_obs = has_final_obs

        raw_env = gym.make(
            env_name,
            num_envs=num_envs,
            reconfiguration_freq=reconfiguration_freq,
            **maniskill_env_kwargs(env_name, env_kwargs),
        )

        if isinstance(raw_env.action_space, gym.spaces.Dict):
            raw_env = FlattenActionSpaceWrapper(raw_env)

        # Torch parity: the training env ends episodes on termination
        # (ignore_terminations=not partial_reset there) and bootstraps them from the
        # final obs; evaluation always runs episodes to the time limit.
        self.env = ManiSkillVectorEnv(
            raw_env,
            num_envs,
            ignore_terminations=ignore_terminations,
            record_metrics=True,
        )

        self.max_episode_steps = gym_utils.find_max_episode_steps_value(self.env)
        self.action_size = self.env.action_space.shape[1]

        # Handle observation shapes (supporting Dict obs common in ManiSkill)
        if isinstance(self.env.observation_space, gym.spaces.Dict):
            self.obs_shape = self.env.observation_space["state"].shape[1:]
        else:
            self.obs_shape = self.env.observation_space.shape[1:]

    def action_space(self, params=None):
        return Box(low=-1.0, high=1.0, shape=(self.action_size,))

    def observation_space(self, params=None):
        return Box(low=-float("inf"), high=float("inf"), shape=self.obs_shape), Box(
            low=-float("inf"), high=float("inf"), shape=self.obs_shape
        )

    def _sync_steps_to_torch(self, jax_steps):
        """Helper to update the physical engine's clock from JAX."""
        # reshape(-1): the caller may hold a (num_seeds, envs_per_seed) view of the flat batch
        torch_steps = to_torch(jax_steps).to(self.env.device).to(torch.long).reshape(-1)
        self.env.unwrapped.elapsed_steps[: self.num_envs] = torch_steps

    def _python_reset(self):
        """This function runs in pure Python, outside of JAX JIT."""
        # ManiSkill manages its own RNG via env_kwargs seed, but we can re-seed here if needed
        obs, info = self.env.reset()
        info = {"steps": info["elapsed_steps"].to(torch.float32), "success": info["success"].to(torch.bool)}
        return to_jax(obs), to_jax(info)

    def _split_obs(self, obs):
        # In ManiSkill, 'state' is usually the proprioception,
        # 'privileged_state' contains the full simulator info.
        if isinstance(obs, dict):
            state_obs = obs["state"]
            critic_obs = obs.get("privileged_state", state_obs) if self.asymmetric_observation else state_obs
            return state_obs, critic_obs
        return obs, obs

    def reset(self, key, params=None) -> ResetReturn:
        # Stepped by the host loop, so the torch engine is driven directly rather than
        # through a callback that XLA could reorder against the rest of the graph.
        obs, info = self._python_reset()

        state_obs, critic_obs = self._split_obs(obs)

        info["truncation"] = jnp.zeros(self.num_envs)

        state = ManiSkillEnvState(
            obs=obs,
            info=info,
            done=jnp.zeros(self.num_envs, dtype=jnp.bool_),
            truncated=jnp.zeros(self.num_envs, dtype=jnp.float32),
            _wrapper=self,
        )
        return state_obs, critic_obs, state

    def _python_step(self, action_jax):
        """Step env via torch outside of JAX JIT."""
        action_torch = to_torch(action_jax)
        obs, reward, terminated, truncated, info = self.env.step(action_torch)

        # The vec env auto-resets and returns the reset obs; the pre-reset terminal obs
        # is exposed only under info["final_observation"] on steps where some env reset.
        # The where-merge produces a fixed-shape bootstrap batch (terminal obs where an
        # env reset, the regular next obs elsewhere), keeping the pytree structure
        # identical on every step.
        if self.has_final_obs and "final_observation" in info:
            reset_mask = torch.logical_or(terminated, truncated)
            final_obs = info["final_observation"]

            def _merge(final, regular):
                mask = reset_mask.reshape(-1, *([1] * (regular.ndim - 1)))
                return torch.where(mask, final, regular)

            if isinstance(obs, dict):
                bootstrap_obs = {k: _merge(final_obs[k], v) for k, v in obs.items()}
            else:
                bootstrap_obs = _merge(final_obs, obs)
        else:
            bootstrap_obs = obs

        # ManiSkill provides elapsed_steps in the info dict but we need it as "steps"
        info = {"steps": info["elapsed_steps"].to(torch.float32), "success": info["success"].to(torch.bool)}

        # Convert everything to JAX
        obs_j = to_jax(obs)
        bootstrap_obs_j = to_jax(bootstrap_obs)
        reward_j = to_jax(reward).flatten()
        term_j = to_jax(terminated).flatten()
        trunc_j = to_jax(truncated).flatten()

        return obs_j, bootstrap_obs_j, reward_j, term_j, trunc_j, to_jax(info)

    def step(self, key, state: ManiSkillEnvState, action: ActionBatch) -> StepReturn:
        obs, bootstrap_obs, reward, terminated, truncated, info = self._python_step(action)

        if self.partial_reset:
            # torch/upstream-REPPO convention: every episode end bootstraps from the
            # terminal obs (surfaced as bootstrap_obs above), so the RL-done stays 0 and
            # the end is flagged as a truncation for the own-step lambda-target bootstrap.
            actual_truncated = jnp.logical_or(terminated, truncated)
            done = jnp.zeros_like(actual_truncated)
        else:
            done = jnp.logical_or(terminated, truncated)
            actual_truncated = jnp.zeros_like(done, dtype=jnp.bool_)

        state_obs, critic_obs = self._split_obs(obs)
        bootstrap_state_obs, bootstrap_critic_obs = self._split_obs(bootstrap_obs)

        info["truncation"] = actual_truncated.astype(jnp.float32)

        new_state = ManiSkillEnvState(
            obs=state_obs, info=info, done=done, truncated=actual_truncated.astype(jnp.float32), _wrapper=self
        )

        # The bootstrap obs travel in the step info so the host boundary stays a
        # 6-tuple; the trainer pops them before the transition is stored.
        step_info = {
            "bootstrap_obs": bootstrap_state_obs,
            "bootstrap_critic_obs": bootstrap_critic_obs,
        }
        return state_obs, critic_obs, new_state, reward, done, step_info


@struct.dataclass
class HumanoidBenchEnvState:
    obs: jnp.ndarray
    info: dict
    done: jnp.ndarray
    truncated: jnp.ndarray
    _wrapper: Any = struct.field(pytree_node=False)  # Reference to sync if needed

    def unwrapped(self):
        return self

    def set_env_state(self, new_state):
        return new_state


def _hb_make_env(env_name, rank, render_mode=None, seed=0):
    """
    Utility function for multiprocessed env.

    :param rank: (int) index of the subprocess
    :param seed: (int) the inital seed for RNG
    """

    if env_name in[
        "h1hand-push-v0",
        "h1-push-v0",
        "h1hand-cube-v0",
        "h1cube-v0",
        "h1hand-basketball-v0",
        "h1-basketball-v0",
        "h1hand-kitchen-v0",
        "h1-kitchen-v0",
    ]:
        max_episode_steps = 500
    else:
        max_episode_steps = 1000

    def _init():
        import humanoid_bench

        env = gym.make(env_name, render_mode=render_mode)
        env = TimeLimit(env, max_episode_steps=max_episode_steps)
        env.unwrapped.seed(seed + rank)

        return env

    return _init


class HumanoidBenchGymnaxWrapper:
    """JAX wrapper for HumanoidBench following the Gymnax/MuJoCo Playground pattern."""

    def __init__(
        self,
        env_name: str,
        num_envs: int,
        render_mode: str = None,
        seed: int = 0,
        num_subproc: int = 1,
    ):
        self.env_name = env_name
        self.num_envs = num_envs
        self.asymmetric_observation = False
        self.seed_val = seed

        self.envs = MixedVecEnv(
            [_hb_make_env(env_name, i, render_mode=render_mode, seed=seed) for i in range(num_envs)],
            num_subproc=num_subproc,
        )

        if env_name in[
            "h1hand-push-v0",
            "h1-push-v0",
            "h1hand-cube-v0",
            "h1cube-v0",
            "h1hand-basketball-v0",
            "h1-basketball-v0",
            "h1hand-kitchen-v0",
            "h1-kitchen-v0",
        ]:
            self.max_episode_steps = 500
        else:
            self.max_episode_steps = 1000

        # Extract spaces directly from the SB3 VecEnv
        self.action_size = self.envs.action_space.shape[-1]
        self.obs_shape = self.envs.observation_space.shape

    def action_space(self, params=None):
        return Box(low=-1.0, high=1.0, shape=(self.action_size,))

    def observation_space(self, params=None):
        return Box(low=-float("inf"), high=float("inf"), shape=self.obs_shape), Box(
            low=-float("inf"), high=float("inf"), shape=self.obs_shape
        )

    def _python_reset(self):
        """This function runs in pure Python, outside of JAX JIT."""
        obs = self.envs.reset()
        info = {"steps": np.zeros(self.num_envs, dtype=np.float32)}
        return obs, info

    def reset(self, key, params=None) -> ResetReturn:
        # Stepped by the host loop, so the SB3 vec env is driven directly rather than
        # through a callback that XLA could reorder against the rest of the graph.
        obs, info = self._python_reset()
        obs = jnp.asarray(obs)
        info = {k: jnp.asarray(v) for k, v in info.items()}

        info["truncation"] = jnp.zeros(self.num_envs, dtype=jnp.float32)

        state = HumanoidBenchEnvState(
            obs=obs,
            info=info,
            done=jnp.zeros(self.num_envs, dtype=jnp.bool_),
            truncated=jnp.zeros(self.num_envs, dtype=jnp.float32),
            _wrapper=self,
        )
        # Note: HumanoidBench does not natively differentiate state & privileged dict like MJX
        # so we pass standard `obs` for both `state_obs` and `critic_obs`.
        return obs, obs, state

    def _python_step(self, action_jax):
        """Step env via SB3 VecEnv pure Python process outside of JAX JIT."""
        action_np = np.asarray(action_jax)
        observations, rewards, dones, raw_infos = self.envs.step(action_np)

        # The SB3 vec env auto-resets, so the obs at an episode end is already the reset
        # obs and the value bootstrap uses it as-is. Known implementation detail copied
        # from upstream REPPO: its wrapper captures raw_infos[i]["terminal_observation"]
        # but its trainer never consumes it either.
        truncateds = np.zeros(self.num_envs, dtype=bool)
        for i in range(self.num_envs):
            if raw_infos[i].get("TimeLimit.truncated", False):
                truncateds[i] = True

        info_out = {"steps": np.zeros(self.num_envs, dtype=np.float32)}

        return (
            observations.astype(np.float32),
            np.asarray(rewards, dtype=np.float32),
            np.asarray(dones, dtype=bool),
            truncateds,
            info_out
        )

    def step(self, key, state: HumanoidBenchEnvState, action: ActionBatch) -> StepReturn:
        obs, reward, done, truncated, info = self._python_step(action)
        obs = jnp.asarray(obs)
        reward, done = jnp.asarray(reward), jnp.asarray(done)
        truncated = jnp.asarray(truncated)
        info = {k: jnp.asarray(v) for k, v in info.items()}

        info["truncation"] = truncated.astype(jnp.float32)

        new_state = HumanoidBenchEnvState(
            obs=obs,
            info=info,
            done=done,
            truncated=truncated.astype(jnp.float32),
            _wrapper=self
        )

        return obs, obs, new_state, reward, done, {}
