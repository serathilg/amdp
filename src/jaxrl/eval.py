from typing import Callable

import jax
from flax import nnx
from flax.struct import PyTreeNode
from gymnax.environments.environment import Environment
from jax import numpy as jnp
from jaxtyping import Array, Float, PRNGKeyArray

from src.jaxrl.types import SACTrainState

# A policy factory turns a train state into `policy(key, obs, critic_obs) -> action`.
Policy = Callable[
    [PRNGKeyArray, Float[Array, "envs obs"], Float[Array, "envs critic_obs"]],
    Float[Array, "envs act"],
]
PolicyFactory = Callable[[SACTrainState], Policy]


def sde_policy_factory(train_state: SACTrainState) -> Policy:
    """Act with a single SDE sample (the stochastic policy)."""
    actor_model = nnx.merge(train_state.actor.graphdef, train_state.actor.params)

    def policy(
        key: PRNGKeyArray,
        obs: Float[Array, "envs obs"],
        critic_obs: Float[Array, "envs critic_obs"],
    ) -> Float[Array, "envs act"]:
        # Only .action is consumed; XLA DCE prunes the unused sample statistics.
        return actor_model.sde_sample(key, obs, stop_grad=True).action

    return policy


def ode_policy_factory(train_state: SACTrainState) -> Policy:
    """Act with the deterministic probability-flow ODE sample (OU variant only)."""
    actor_model = nnx.merge(train_state.actor.graphdef, train_state.actor.params)

    def policy(
        key: PRNGKeyArray,
        obs: Float[Array, "envs obs"],
        critic_obs: Float[Array, "envs critic_obs"],
    ) -> Float[Array, "envs act"]:
        return actor_model.ode_sample(key, obs, stop_grad=True)

    return policy


def make_best_of_n_policy_factory(best_of_n: int) -> PolicyFactory:
    """Act with the argmax-Q action among `best_of_n` SDE samples."""

    def factory(train_state: SACTrainState) -> Policy:
        actor_model = nnx.merge(train_state.actor.graphdef, train_state.actor.params)
        critic_model = nnx.merge(train_state.critic.graphdef, train_state.critic.params)

        def single_sample_and_Q(key_single_sample, obs, critic_obs):
            # Only .action is consumed; XLA DCE prunes the unused sample statistics.
            action = actor_model.sde_sample(key_single_sample, obs, stop_grad=True).action
            q_val = critic_model.critic(critic_obs, action)
            return action, q_val

        def policy(
            key: PRNGKeyArray,
            obs: Float[Array, "envs obs"],
            critic_obs: Float[Array, "envs critic_obs"],
        ) -> Float[Array, "envs act"]:
            sample_keys = jax.random.split(key, (int(best_of_n),))
            batched_sde_sample_and_Q = jax.vmap(single_sample_and_Q, in_axes=(0, None, None))
            actions, q_vals = batched_sde_sample_and_Q(sample_keys, obs, critic_obs)
            return actions[jnp.argmax(q_vals, axis=0), jnp.arange(q_vals.shape[1])]

        return policy

    return factory


def aggregate_eval_infos(
    infos: dict, eval_first_episode_only: bool
) -> dict[str, Float[Array, ""]]:
    """Reduce the stacked per-step infos of an evaluation rollout to episode metrics."""
    dones = infos["returned_episode"]
    first_done_indices = jnp.argmax(dones, axis=0)
    # Mask if no env ever finished (where argmax returns 0)
    env_has_done = jnp.any(dones, axis=0)
    num_envs = dones.shape[1]

    # Compute unbiased mean/std of first episode to avoid size-bias/inspection-paradox
    def first_dones_only_agg(metric, agg_fun):
        first_metric = metric[first_done_indices, jnp.arange(num_envs)]
        return agg_fun(first_metric, where=env_has_done)

    def all_done_agg(metric, agg_fun):
        return agg_fun(metric, where=dones)

    agg = first_dones_only_agg if eval_first_episode_only else all_done_agg

    return {
        "episode_return": agg(infos["returned_episode_returns"], jnp.mean),
        "episode_return_std": agg(infos["returned_episode_returns"], jnp.std),
        "episode_length": agg(infos["returned_episode_lengths"], jnp.mean),
        "episode_length_std": agg(infos["returned_episode_lengths"], jnp.std),
        "episode_success": agg(infos["returned_episode_successes"], jnp.mean),
        "episode_success_std": agg(infos["returned_episode_successes"], jnp.std),
        "num_episodes": agg(infos["returned_episode"], jnp.sum),
    }


def make_eval_fn(
    env: Environment,
    max_episode_steps: int,
    policy_factory: PolicyFactory,
    eval_first_episode_only: bool = False,
) -> Callable[[PRNGKeyArray, SACTrainState, PyTreeNode | None], dict[str, Float[Array, ""]]]:
    """Evaluation rollout for a JAX-native environment: one `lax.scan` over the episode.

    The whole function is traced by the caller, so `env.step` is staged into the graph.
    """

    def evaluation_fn(
        key: PRNGKeyArray,
        train_state: SACTrainState,
        norm_state: PyTreeNode | None,
    ) -> dict[str, Float[Array, ""]]:
        policy = policy_factory(train_state)

        def step_env(carry, _):
            key, env_state, obs, critic_obs = carry
            key, act_key, env_key = jax.random.split(key, 3)
            action = policy(act_key, obs, critic_obs)

            step_key = jax.random.split(env_key, env.num_envs)
            obs, critic_obs, env_state, reward, done, info = env.step(
                step_key, env_state, action, update_stats=False
            )
            return (key, env_state, obs, critic_obs), info

        key, init_key = jax.random.split(key)
        init_key = jax.random.split(init_key, env.num_envs)
        obs, critic_obs, env_state = env.reset(init_key, norm_state)

        key, env_key = jax.random.split(key)
        _, infos = jax.lax.scan(
            f=step_env,
            init=(key, env_state, obs, critic_obs),
            xs=None,
            length=max_episode_steps,
        )

        return aggregate_eval_infos(infos, eval_first_episode_only)

    return evaluation_fn


def make_host_eval_fn(
    env: Environment,
    max_episode_steps: int,
    policy_factory: PolicyFactory,
    eval_first_episode_only: bool = False,
    num_seeds: int = 1,
    num_envs: int | None = None,
    normalize: bool = True,
) -> Callable[[PRNGKeyArray, SACTrainState, PyTreeNode | None], dict[str, Float[Array, "seeds"]]]:
    """Evaluation rollout for a host-stepped environment: a Python loop over the episode.

    `env` is the raw vec env batching all seeds (num_seeds * num_envs physical envs);
    every traced piece — policy, wrapper statistics math, aggregation — is vmapped over
    the seed axis, and the boundary reshapes to the flat physical batch. Keys and the
    train state carry the seed axis, exactly as in the JAX-native path.
    """
    from src.env_utils.jax_wrappers import wrap_reset_stats, wrap_step_stats

    jit_policy = jax.jit(
        jax.vmap(
            lambda train_state, key, obs, critic_obs: policy_factory(train_state)(
                key, obs, critic_obs
            )
        )
    )
    jit_aggregate = jax.jit(
        jax.vmap(
            lambda infos: aggregate_eval_infos(infos, eval_first_episode_only),
            in_axes=1,  # infos are stacked as (steps, seeds, envs)
        )
    )
    jit_reset_stats = jax.jit(
        jax.vmap(
            lambda obs, critic_obs, raw_state, norm_state: wrap_reset_stats(
                obs, critic_obs, raw_state, num_envs, norm_state, normalize
            )
        )
    )
    jit_reset_stats_no_params = jax.jit(
        jax.vmap(
            lambda obs, critic_obs, raw_state: wrap_reset_stats(
                obs, critic_obs, raw_state, num_envs, None, normalize
            )
        )
    )
    jit_step_stats = jax.jit(
        jax.vmap(
            lambda state, obs, critic_obs, raw_state, reward, done, info: wrap_step_stats(
                state,
                obs,
                critic_obs,
                raw_state,
                reward,
                done,
                info,
                update_stats=False,
                normalize=normalize,
            )
        )
    )
    jit_advance_keys = jax.jit(jax.vmap(lambda k: jax.random.split(k, 3)))

    def unflatten_seeds(tree):
        return jax.tree.map(
            lambda x: x.reshape(num_seeds, num_envs, *x.shape[1:]), tree
        )

    jit_consume_key = jax.jit(jax.vmap(lambda k: jax.random.split(k)[0]))

    def evaluation_fn(
        keys: PRNGKeyArray,
        train_state: SACTrainState,
        norm_state: PyTreeNode | None,
    ) -> dict[str, Float[Array, "seeds"]]:
        # Mirror the JAX-native derivation: one split consumed by the env-reset keys
        # (unused by a host-stepped env) and one by the env-step keys.
        keys = jit_consume_key(jit_consume_key(keys))

        obs, critic_obs, raw_state = env.reset(None)
        obs, critic_obs, raw_state = unflatten_seeds((obs, critic_obs, raw_state))
        if norm_state is not None:
            obs, critic_obs, env_state = jit_reset_stats(
                obs, critic_obs, raw_state, norm_state
            )
        else:
            obs, critic_obs, env_state = jit_reset_stats_no_params(
                obs, critic_obs, raw_state
            )

        infos = []
        for _ in range(max_episode_steps):
            keys, act_keys, _env_keys = jit_advance_keys(keys).transpose(1, 0, 2)
            action = jit_policy(train_state, act_keys, obs, critic_obs)

            # ClipAction's range; the raw env is stepped without the wrapper chain.
            action = jnp.clip(action, -0.999, 0.999)
            flat_action = action.reshape(num_seeds * num_envs, *action.shape[2:])
            raw = env.step(None, None, flat_action)
            # Evaluation has no value bootstrap; drop the terminal-obs batch some host
            # envs (ManiSkill) ship in the step info before it is traced or stacked.
            raw[-1].pop("bootstrap_obs", None)
            raw[-1].pop("bootstrap_critic_obs", None)

            obs, critic_obs, env_state, reward, done, info = jit_step_stats(
                env_state, *unflatten_seeds(raw)
            )
            infos.append(info)

        infos = jax.tree.map(lambda *xs: jnp.stack(xs), *infos)
        return jit_aggregate(infos)

    return evaluation_fn
