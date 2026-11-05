import logging
import time
import traceback
from functools import partial
from typing import Callable

import hydra
import jax
import numpy as np
import optax
import optuna
from flax import nnx
from flax.struct import PyTreeNode
from gymnax.environments.environment import Environment, EnvParams
from jax import numpy as jnp
from jaxtyping import Array, Float, PRNGKeyArray
from omegaconf import DictConfig, OmegaConf

import wandb
from src.env_utils.jax_wrappers import (
    ClipAction,
    HumanoidBenchGymnaxWrapper,
    LogWrapper,
    ManiSkillGymnaxWrapper,
    MjxGymnaxWrapper,
    NormalizeVec,
    normalize_obs,
    wrap_reset_stats,
    wrap_step_stats,
)
from src.jaxrl import utils
from src.jaxrl.eval import (
    make_best_of_n_policy_factory,
    make_eval_fn,
    make_host_eval_fn,
    sde_policy_factory,
)
from src.jaxrl.lagrangian_utils import (
    compute_entropy_via_importance_sampling,
    find_optimum_kl_lagrangian,
)
from src.jaxrl.types import ReppoConfigBase, SACTrainState, Transition
from src.jaxrl.utils import _get_optuna_type, compute_reverse_ess
from src.networks.jax_models import (
    CategoricalCriticNetwork,
    CriticNetwork,
)
from src.networks.amdp.jax_control_net import ControlNetwork, ResidualControlNetwork

from src.networks.amdp.jax_policy import (
    BrownianSDE,
    DiffusionActor,
)
from src.networks.amdp.types import SDESample

logging.basicConfig(level=logging.INFO)


class ReppoConfig(ReppoConfigBase):
    best_of_n_values: list = None


def make_init(
    cfg: ReppoConfig,
    env: Environment,
    env_params: EnvParams = None,
) -> tuple[Callable, Callable, Callable, Callable]:
    # Built outside init so that its schedule constants stay concrete: under jit they
    # would be staged into the trace and escape it inside the actor's graphdef.
    noise_schedule = hydra.utils.call(cfg.diffusion.noise_schedule)
    sde_integrator = hydra.utils.get_method(cfg.diffusion.sde_integrator)

    def init_models(key: PRNGKeyArray):
        """Build the actor/critic train states; no environment interaction."""
        key, model_key = jax.random.split(key)
        obs_dim=env.observation_space(env_params)[0].shape[0]
        critic_obs_dim=env.observation_space(env_params)[1].shape[0]
        action_dim=env.action_space(env_params).shape[0]

        if not cfg.diffusion.score_model.residual:
            forward_model: nnx.Module = ControlNetwork(
                action_dim=action_dim,
                observation_dim=obs_dim,
                num_layers=cfg.diffusion.score_model.num_layers,
                num_hid=cfg.diffusion.score_model.num_hid,
                time_mode=cfg.diffusion.score_model.time_mode,
                time_mlp_input=cfg.diffusion.score_model.time_mlp_input,
                num_time_fourier=cfg.diffusion.score_model.num_time_fourier,
                time_fourier_range_min=cfg.diffusion.score_model.time_fourier_range_min,
                time_fourier_range_max=cfg.diffusion.score_model.time_fourier_range_max,
                num_time_hid=cfg.diffusion.score_model.num_time_hid,
                num_time_out=cfg.diffusion.score_model.num_time_out,
                action_mode=cfg.diffusion.score_model.action_mode,
                action_mlp_input=cfg.diffusion.score_model.action_mlp_input,
                num_action_fourier=cfg.diffusion.score_model.num_action_fourier,
                action_fourier_range_min=cfg.diffusion.score_model.action_fourier_range_min,
                action_fourier_range_max=cfg.diffusion.score_model.action_fourier_range_max,
                num_action_hid=cfg.diffusion.score_model.num_action_hid,
                num_action_out=cfg.diffusion.score_model.num_action_out,
                outer_clip=cfg.diffusion.score_model.outer_clip,
                inner_clip=cfg.diffusion.score_model.inner_clip,
                weight_init=cfg.diffusion.score_model.weight_init,
                bias_init=cfg.diffusion.score_model.bias_init,
                layer_norm=cfg.diffusion.score_model.layer_norm,
                layer_norm_type=cfg.diffusion.score_model.layer_norm_type,
                rngs=nnx.Rngs(model_key),
            )
        else:
            forward_model: nnx.Module = ResidualControlNetwork(
                action_dim=action_dim,
                observation_dim=obs_dim,
                stream_dim=cfg.diffusion.score_model.res_stream_dim,
                num_blocks=cfg.diffusion.score_model.num_res_blocks,
                cond_mode=cfg.diffusion.score_model.res_condition_mode,
                context_dim=cfg.diffusion.score_model.res_adaln_context_dim,
                activation=cfg.diffusion.score_model.res_block_activation,
                layer_norm_type=cfg.diffusion.score_model.layer_norm_type,
                simba_shift=False,
                time_mode=cfg.diffusion.score_model.time_mode,
                time_mlp_input=cfg.diffusion.score_model.time_mlp_input,
                num_time_fourier=cfg.diffusion.score_model.num_time_fourier,
                time_fourier_range_min=cfg.diffusion.score_model.time_fourier_range_min,
                time_fourier_range_max=cfg.diffusion.score_model.time_fourier_range_max,
                num_time_hid=cfg.diffusion.score_model.num_time_hid,
                num_time_out=cfg.diffusion.score_model.num_time_out,
                action_mode=cfg.diffusion.score_model.action_mode,
                action_mlp_input=cfg.diffusion.score_model.action_mlp_input,
                num_action_fourier=cfg.diffusion.score_model.num_action_fourier,
                action_fourier_range_min=cfg.diffusion.score_model.action_fourier_range_min,
                action_fourier_range_max=cfg.diffusion.score_model.action_fourier_range_max,
                num_action_hid=cfg.diffusion.score_model.num_action_hid,
                num_action_out=cfg.diffusion.score_model.num_action_out,
                outer_clip=cfg.diffusion.score_model.outer_clip,
                weight_init=cfg.diffusion.score_model.weight_init,
                bias_init=cfg.diffusion.score_model.bias_init,
                rngs=nnx.Rngs(model_key),
            )

        diffusion_model = BrownianSDE(
            action_dim=action_dim,
            observation_dim=obs_dim,
            fwd_model=forward_model,
            diff_steps=cfg.diffusion.diff_steps,
            scheduler=noise_schedule,
            rngs=nnx.Rngs(model_key),
        )

        actor_networks = DiffusionActor(
            action_dim=action_dim,
            observation_dim=obs_dim,
            diffusion_model=diffusion_model,
            kl_start=cfg.kl_start,
            ent_start=cfg.ent_start,
            sde_integrator=sde_integrator,
            kl_bound=cfg.kl_bound,
            squashing=cfg.diffusion.squashing,
        )

        actor_target_networks = DiffusionActor(
            action_dim=action_dim,
            observation_dim=obs_dim,
            diffusion_model=diffusion_model,
            kl_start=cfg.kl_start,
            ent_start=cfg.ent_start,
            sde_integrator=sde_integrator,
            kl_bound=cfg.kl_bound,
            squashing=cfg.diffusion.squashing,
        )

        if cfg.hl_gauss:
            critic_networks: nnx.Module = CategoricalCriticNetwork(
                obs_dim=critic_obs_dim,
                action_dim=action_dim,
                hidden_dim=cfg.critic_hidden_dim,
                num_bins=cfg.num_bins,
                vmin=cfg.vmin,
                vmax=cfg.vmax,
                use_norm=cfg.use_critic_norm,
                encoder_layers=cfg.num_critic_encoder_layers,
                use_simplical_embedding=cfg.use_simplical_embedding,
                head_layers=cfg.num_critic_head_layers,
                pred_layers=cfg.num_critic_pred_layers,
                use_skip=cfg.use_critic_skip,
                rngs=nnx.Rngs(model_key),
            )
            critic_target_networks: nnx.Module = CategoricalCriticNetwork(
                obs_dim=critic_obs_dim,
                action_dim=action_dim,
                hidden_dim=cfg.critic_hidden_dim,
                num_bins=cfg.num_bins,
                vmin=cfg.vmin,
                vmax=cfg.vmax,
                use_norm=cfg.use_critic_norm,
                encoder_layers=cfg.num_critic_encoder_layers,
                use_simplical_embedding=cfg.use_simplical_embedding,
                head_layers=cfg.num_critic_head_layers,
                pred_layers=cfg.num_critic_pred_layers,
                use_skip=cfg.use_critic_skip,
                rngs=nnx.Rngs(model_key),
            )
        else:
            critic_networks: nnx.Module = CriticNetwork(
                obs_dim=critic_obs_dim,
                action_dim=action_dim,
                hidden_dim=cfg.critic_hidden_dim,
                use_norm=cfg.use_critic_norm,
                encoder_layers=cfg.num_critic_encoder_layers,
                use_simplical_embedding=cfg.use_simplical_embedding,
                head_layers=cfg.num_critic_head_layers,
                pred_layers=cfg.num_critic_pred_layers,
                use_skip=cfg.use_critic_skip,
                rngs=nnx.Rngs(model_key),
            )
            critic_target_networks: nnx.Module = CriticNetwork(
                obs_dim=critic_obs_dim,
                action_dim=action_dim,
                hidden_dim=cfg.critic_hidden_dim,
                use_norm=cfg.use_critic_norm,
                encoder_layers=cfg.num_critic_encoder_layers,
                use_simplical_embedding=cfg.use_simplical_embedding,
                head_layers=cfg.num_critic_head_layers,
                pred_layers=cfg.num_critic_pred_layers,
                use_skip=cfg.use_critic_skip,
                rngs=nnx.Rngs(model_key),
            )

        if not cfg.anneal_lr:
            lr = cfg.lr
        else:
            num_iterations = cfg.total_time_steps // cfg.num_steps // cfg.num_envs
            num_updates = num_iterations * cfg.num_epochs * cfg.num_mini_batches
            lr = optax.linear_schedule(cfg.lr, 0, num_updates)

        actor_optimizer = optax.adam(lr)

        critic_optimizer = optax.chain(
            optax.clip_by_global_norm(cfg.max_grad_norm) if cfg.max_grad_norm else optax.identity(),
            optax.adam(lr)
        )

        actor_trainstate = nnx.TrainState.create(
            graphdef=nnx.graphdef(actor_networks),
            params=nnx.state(actor_networks),
            tx=actor_optimizer,
        )
        actor_target_trainstate = nnx.TrainState.create(
            graphdef=nnx.graphdef(actor_target_networks),
            params=nnx.state(actor_target_networks),
            tx=optax.set_to_zero(),
        )
        critic_trainstate = nnx.TrainState.create(
            graphdef=nnx.graphdef(critic_networks),
            params=nnx.state(critic_networks),
            tx=critic_optimizer,
        )
        critic_target_trainstate = nnx.TrainState.create(
            graphdef=nnx.graphdef(critic_target_networks),
            params=nnx.state(critic_target_networks),
            tx=optax.set_to_zero(),
        )

        actor_param_count = utils.count_params(actor_trainstate.params)
        critic_param_count = utils.count_params(critic_trainstate.params)
        
        print(f"Actor parameters: {actor_param_count:,}")
        print(f"Critic parameters: {critic_param_count:,}")
        print(f"Total parameters: {actor_param_count + critic_param_count:,}")

        return (
            key,
            actor_trainstate,
            actor_target_trainstate,
            critic_trainstate,
            critic_target_trainstate,
        )

    def init_env(
        key: PRNGKeyArray,
    ) -> tuple[Float[Array, "envs obs"], Float[Array, "envs critic_obs"], PyTreeNode]:
        """Reset the environment. Runs inside the trace for JAX-native envs, on the host
        otherwise, so it is kept apart from the model init."""
        key, env_key = jax.random.split(key)
        env_key = jax.random.split(env_key, cfg.num_envs)
        obs, critic_obs, env_state = env.reset(key=env_key, params=env_params)

        if cfg.use_staggered_resets:
            # Deterministic group stagger of the initial elapsed steps (torch parity):
            # desynchronizes episode ends across env groups. Opt-in, off by default.
            _env_state = env_state.unwrapped()
            _env_state.info["steps"] = utils.staggered_reset_offsets(
                cfg.num_envs,
                cfg.max_episode_steps,
                cfg.num_steps,
                cfg.num_stagger_blocks,
            )
            env_state = env_state.set_env_state(_env_state)
        else:
            # randomize initial time step to prevent all envs stepping in tandem
            _env_state = env_state.unwrapped()
            key, randomize_steps_key = jax.random.split(key)
            _env_state.info["steps"] = jax.random.randint(
                randomize_steps_key,
                _env_state.info["steps"].shape,
                0,
                cfg.max_episode_steps,
            ).astype(jnp.float32)
            env_state.set_env_state(_env_state)

        return obs, critic_obs, env_state

    def assemble(
        trainstates: tuple,
        obs: Float[Array, "envs obs"],
        critic_obs: Float[Array, "envs critic_obs"],
        env_state: PyTreeNode,
    ) -> SACTrainState:
        actor, actor_target, critic, critic_target = trainstates
        return SACTrainState(
            actor=actor,
            actor_target=actor_target,
            critic=critic,
            critic_target=critic_target,
            iteration=0,
            time_steps=0,
            last_env_state=env_state,
            last_obs=obs,
            last_critic_obs=critic_obs,
        )

    def init(key: PRNGKeyArray) -> SACTrainState:
        key, *trainstates = init_models(key)
        obs, critic_obs, env_state = init_env(key)
        return assemble(trainstates, obs, critic_obs, env_state)

    return init, init_models, init_env, assemble


def make_train_fn(
    cfg: ReppoConfig,
    env: Environment,
    eval_env: Environment,
    env_params: EnvParams = None,
    log_callback: Callable[[SACTrainState, dict[str, jax.Array]], None] | None = None,
    num_seeds: int = 1,
    jax_native: bool = True,
    partial_reset: bool = False,
):
    """
    Create training function with support for Best-of-N evaluation.

    Args:
        cfg: Configuration
        env: Environment
        eval_env: Evaluation Environment
        env_params: Environment parameters
        log_callback: Logging callback
        num_seeds: Number of seeds
        jax_native: Whether the environment steps inside the traced graph (see `run`).
            Selects the lambda-target and critic-mask convention: JAX-native envs keep
            the brax-lineage carry semantics, host-stepped envs replicate the
            torch/upstream-REPPO semantics.
        partial_reset: Host-path only (torch parity): episode ends bootstrap from the
            terminal obs and nothing is masked from the critic loss.

    Shape convention: every closure below is annotated with per-seed shapes
    ("envs obs" etc.); callers vmap them over a leading seed axis, so the arrays
    at run time carry (seeds, ...). Annotations are documentation, not
    runtime-checked.
    """
    if jax_native:
        # The wrapper chain is traced along with the env. A host-stepped env stays raw:
        # its rollout applies the same statistics math per seed via the pure functions
        # (wrap_reset_stats / wrap_step_stats), outside the wrapper classes.
        env = LogWrapper(env, cfg.num_envs)
        eval_env = LogWrapper(eval_env, cfg.num_envs)
        env = ClipAction(env)
        eval_env = ClipAction(eval_env)
        env = NormalizeVec(env, enable=cfg.normalize_env)
        eval_env = NormalizeVec(eval_env, enable=cfg.normalize_env)
        build_eval_fn = make_eval_fn
    else:
        build_eval_fn = partial(
            make_host_eval_fn,
            num_seeds=num_seeds,
            num_envs=cfg.num_envs,
            normalize=cfg.normalize_env,
        )
    sde_eval_fn = build_eval_fn(
        eval_env,
        cfg.max_episode_steps,
        sde_policy_factory,
        eval_first_episode_only=cfg.eval_first_episode_only,
    )
    best_of_n_eval_fns = {
        int(n): build_eval_fn(
            eval_env,
            cfg.max_episode_steps,
            make_best_of_n_policy_factory(int(n)),
            eval_first_episode_only=cfg.eval_first_episode_only,
        )
        for n in (getattr(cfg, "best_of_n_values", None) or [])
    }
    action_size_target = jnp.prod(jnp.array(env.action_space(env_params).shape)) * cfg.ent_target_mult

    def sample_action(
        train_state: SACTrainState, obs: Float[Array, "envs obs"], key: PRNGKeyArray
    ) -> SDESample:
        """Draw the policy action for one env step (the pre-`env.step` half of the body)."""
        actor_model = nnx.merge(train_state.actor.graphdef, train_state.actor.params)
        return actor_model.sde_sample(key, obs, stop_grad=True)

    def make_transition(
        train_state: SACTrainState,
        obs: Float[Array, "envs obs"],
        critic_obs: Float[Array, "envs critic_obs"],
        sample: SDESample,
        next_obs: Float[Array, "envs obs"],
        next_critic_obs: Float[Array, "envs critic_obs"],
        reward: Float[Array, "envs"],
        done: Float[Array, "envs"],
        truncated: Float[Array, "envs"],
        info: dict,
        key: PRNGKeyArray,
        bootstrap_obs: Float[Array, "envs obs"] | None = None,
        bootstrap_critic_obs: Float[Array, "envs critic_obs"] | None = None,
    ) -> Transition:
        """Assemble one transition (the post-`env.step` half of the body).

        `bootstrap_obs`/`bootstrap_critic_obs` feed the next-state value, embedding, and
        entropy sample: the pre-reset terminal obs where a host env auto-reset this step
        (torch parity), the regular next obs otherwise (the default).
        """
        if bootstrap_obs is None:
            bootstrap_obs = next_obs
        if bootstrap_critic_obs is None:
            bootstrap_critic_obs = next_critic_obs
        actor_model = nnx.merge(train_state.actor.graphdef, train_state.actor.params)
        critic_model = nnx.merge(
            train_state.critic_target.graphdef, train_state.critic_target.params
        )

        action = jnp.clip(sample.action, -0.999, 0.999)
        importance_weight = jnp.zeros((cfg.num_envs,))

        # compute next state embedding and value
        next_sample = actor_model.sde_sample(key, bootstrap_obs, stop_grad=True)
        next_action = jax.lax.stop_gradient(next_sample.action)
        next_log_prob = -next_sample.log_weight
        next_log_prob = next_log_prob.sum(-1)
        next_emb, _, _, value = critic_model.forward(bootstrap_critic_obs, next_action)
        soft_reward = reward - cfg.gamma * next_log_prob.squeeze() * actor_model.temperature()

        return Transition(
            obs=obs,
            critic_obs=critic_obs,
            prior_action=sample.prior_action,
            action=action,
            action_unsquashed=sample.action_unsquashed,
            squashing_correction_grad=sample.squashing_correction_grad,
            squashing_correction_val=sample.squashing_correction_val,
            next_emb=next_emb,
            reward=reward,
            soft_reward=soft_reward,
            value=value,
            done=done,
            truncated=truncated,
            info=info,
            importance_weight=importance_weight,
            log_weights=sample.log_weight,
            Q_value=jnp.zeros_like(value),
            Q_score=jnp.zeros_like(action),
            log_path_weight_deterministic=sample.log_path_weight_deterministic,
            log_path_weight_stochastic=sample.log_path_weight_stochastic,
            log_p_T_ref=sample.log_p_T_ref,
        )

    def advance_train_state(
        train_state: SACTrainState,
        env_state: PyTreeNode,
        obs: Float[Array, "envs obs"],
        critic_obs: Float[Array, "envs critic_obs"],
    ) -> SACTrainState:
        return train_state.replace(
            last_env_state=env_state,
            last_obs=obs,
            last_critic_obs=critic_obs,
            time_steps=train_state.time_steps + cfg.num_steps * cfg.num_envs,
        )

    def collect_rollout(
        key: PRNGKeyArray, train_state: SACTrainState
    ) -> tuple[Transition, SACTrainState]:
        """Rollout for a JAX-native env: the whole body is traced into one `lax.scan`."""

        def step_env(carry, _) -> tuple[tuple, Transition]:
            key, env_state, train_state, obs, critic_obs = carry
            key, act_key, step_key = jax.random.split(key, 3)
            step_key = jax.random.split(step_key, cfg.num_envs)

            sample = sample_action(train_state, obs, act_key)

            next_obs, next_critic_obs, next_env_state, reward, done, info = env.step(
                step_key, env_state, sample.action
            )

            key, next_act_key = jax.random.split(key)
            transition = make_transition(
                train_state,
                obs,
                critic_obs,
                sample,
                next_obs,
                next_critic_obs,
                reward,
                done,
                next_env_state.truncated,
                info,
                next_act_key,
            )
            return (
                key,
                next_env_state,
                train_state,
                next_obs,
                next_critic_obs,
            ), transition

        rollout_state, transitions = jax.lax.scan(
            f=step_env,
            init=(
                key,
                train_state.last_env_state,
                train_state,
                train_state.last_obs,
                train_state.last_critic_obs,
            ),
            length=cfg.num_steps,
        )
        _, last_env_state, train_state, last_obs, last_critic_obs = rollout_state

        return transitions, advance_train_state(
            train_state, last_env_state, last_obs, last_critic_obs
        )

    def unflatten_seeds(tree):
        """(num_seeds*num_envs, ...) -> (num_seeds, num_envs, ...) on every array leaf.

        The physical host-stepped env batches all seeds; the traced phases are vmapped
        over the seed axis. An identity view for a single seed.
        """
        return jax.tree.map(
            lambda x: x.reshape(num_seeds, cfg.num_envs, *x.shape[1:]), tree
        )

    def host_pre_step(
        train_state: SACTrainState, obs: Float[Array, "envs obs"], key: PRNGKeyArray
    ) -> tuple[PRNGKeyArray, PRNGKeyArray, SDESample]:
        """Per-seed key derivation and action sampling (the pre-`env.step` half)."""
        key, act_key, _env_key = jax.random.split(key, 3)
        sample = sample_action(train_state, obs, act_key)
        key, next_act_key = jax.random.split(key)
        return key, next_act_key, sample

    def host_post_step(
        train_state: SACTrainState,
        env_state: PyTreeNode,
        obs: Float[Array, "envs obs"],
        critic_obs: Float[Array, "envs critic_obs"],
        sample: SDESample,
        raw: tuple,
        next_act_key: PRNGKeyArray,
    ) -> tuple[
        Float[Array, "envs obs"], Float[Array, "envs critic_obs"], PyTreeNode, Transition
    ]:
        """Per-seed statistics update and transition assembly (the post-step half)."""
        raw_obs, raw_critic_obs, raw_state, reward, done, info = raw
        # A host env that surfaces the pre-reset terminal obs (ManiSkill) ships it in the
        # step info; popped here so it feeds the bootstrap without entering Transition.info.
        raw_bootstrap_obs = info.pop("bootstrap_obs", raw_obs)
        raw_bootstrap_critic_obs = info.pop("bootstrap_critic_obs", raw_critic_obs)
        next_obs, next_critic_obs, next_env_state, reward, done, info = wrap_step_stats(
            env_state,
            raw_obs,
            raw_critic_obs,
            raw_state,
            reward,
            done,
            info,
            update_stats=True,
            normalize=cfg.normalize_env,
        )
        if cfg.normalize_env:
            bootstrap_obs = normalize_obs(
                raw_bootstrap_obs, next_env_state.mean, next_env_state.var
            )
            bootstrap_critic_obs = normalize_obs(
                raw_bootstrap_critic_obs,
                next_env_state.critic_mean,
                next_env_state.critic_var,
            )
        else:
            bootstrap_obs = raw_bootstrap_obs
            bootstrap_critic_obs = raw_bootstrap_critic_obs
        transition = make_transition(
            train_state,
            obs,
            critic_obs,
            sample,
            next_obs,
            next_critic_obs,
            reward,
            done,
            next_env_state.truncated,
            info,
            next_act_key,
            bootstrap_obs=bootstrap_obs,
            bootstrap_critic_obs=bootstrap_critic_obs,
        )
        return next_obs, next_critic_obs, next_env_state, transition

    jit_host_pre_step = jax.jit(jax.vmap(host_pre_step))
    jit_host_post_step = jax.jit(jax.vmap(host_post_step))

    def host_collect_rollout(
        keys: PRNGKeyArray, train_state: SACTrainState
    ) -> tuple[Transition, SACTrainState]:
        """Rollout for a host-stepped env: `env.step` runs eagerly between two jitted halves.

        The env never enters the traced graph, so it cannot be reordered against it.
        Everything traced is vmapped over the seed axis; the raw env sees the flat
        (num_seeds*num_envs) batch. Key derivation mirrors the JAX-native `step_env`.
        """
        env_state = train_state.last_env_state
        obs, critic_obs = train_state.last_obs, train_state.last_critic_obs

        transitions = []
        for _ in range(cfg.num_steps):
            keys, next_act_keys, sample = jit_host_pre_step(train_state, obs, keys)

            # ClipAction's range; the raw env is stepped without the wrapper chain.
            action = jnp.clip(sample.action, -0.999, 0.999)
            flat_action = action.reshape(num_seeds * cfg.num_envs, *action.shape[2:])
            # The raw wrappers are stateful python objects; the state argument is unused.
            raw = env.step(None, None, flat_action)

            obs, critic_obs, env_state, transition = jit_host_post_step(
                train_state,
                env_state,
                obs,
                critic_obs,
                sample,
                unflatten_seeds(raw),
                next_act_keys,
            )
            transitions.append(transition)

        # (steps, seeds, envs, ...) -> (seeds, steps, envs, ...), as the vmapped
        # JAX-native rollout produces it.
        transitions = jax.tree.map(lambda *xs: jnp.stack(xs, axis=1), *transitions)

        return transitions, advance_train_state(train_state, env_state, obs, critic_obs)

    def learn_step(
        key: PRNGKeyArray, train_state: SACTrainState, batch: Transition
    ) -> tuple[SACTrainState, dict[str, Float[Array, ""]]]:
        # compute n-step lambda estimates (host envs use the torch-lineage own-step
        # truncation bootstrap; see compute_lambda_targets)
        target_values = utils.compute_lambda_targets(
            batch, cfg.gamma, cfg.lmbda, bootstrap_at_own_step=not jax_native
        )
        # Reshape data to (num_steps * num_envs, ...)
        data = (batch, target_values)
        data = jax.tree.map(
            lambda x: x.reshape((cfg.num_steps * cfg.num_envs, *x.shape[2:])), data
        )

        train_state = train_state.replace(
            actor_target=train_state.actor_target.replace(
                params=train_state.actor.params
            ),
        )
        actor_target_model = nnx.merge(
            train_state.actor_target.graphdef, train_state.actor_target.params
        )




        def update_critic(train_state, key) -> tuple[SACTrainState, dict[str, jax.Array]]:
            def minibatch_update(carry, indices):
                idx, train_state = carry
                # Sample data at indices from the batch
                minibatch, target_values = jax.tree.map(
                    lambda x: jnp.take(x, indices, axis=0), data
                )

                def critic_loss_fn(params):
                    critic_model = nnx.merge(train_state.critic.graphdef, params)
                    critic_pred = critic_model.critic_cat(
                        minibatch.critic_obs, minibatch.action
                    ).squeeze()
                    if cfg.hl_gauss:
                        target_cat = jax.vmap(
                            utils.hl_gauss, in_axes=(0, None, None, None)
                        )(target_values, cfg.num_bins, cfg.vmin, cfg.vmax)
                        critic_update_loss = optax.softmax_cross_entropy(
                            critic_pred, target_cat
                        )
                    else:
                        critic_update_loss = optax.squared_error(
                            critic_pred.reshape(-1, 1),
                            target_values.reshape(-1, 1),
                        )

                    # Aux loss
                    _, pred, pred_rew, value = critic_model.forward(
                        minibatch.critic_obs, minibatch.action
                    )
                    aux_loss = optax.squared_error(pred,  minibatch.next_emb)
                    if jax_native:
                        # JAX-lineage masks: the aux target crosses the reset on any
                        # episode end (same-step autoreset), and the critic target of a
                        # truncated step cannot be formed without the terminal obs.
                        aux_loss = jnp.mean(
                            (1 - minibatch.done.reshape(-1, 1))
                            * aux_loss, axis=-1)
                        loss_mask = 1.0 - minibatch.truncated
                    else:
                        # torch/upstream-REPPO masks: under partial_reset every episode
                        # end bootstraps from a valid terminal obs, so nothing is
                        # masked; otherwise truncated steps are dropped (HumanoidBench).
                        aux_loss = jnp.mean(aux_loss, axis=-1)
                        loss_mask = (
                            jnp.ones_like(minibatch.truncated)
                            if partial_reset
                            else 1.0 - minibatch.truncated
                        )

                    # compute l2 error for logging
                    critic_loss = optax.squared_error(
                        value,
                        target_values,
                    )
                    critic_loss = jnp.mean(critic_loss)
                    loss = jnp.mean(
                        loss_mask
                        * (critic_update_loss + cfg.aux_loss_mult * aux_loss)
                    )
                    # log critic parameters norm
                    metrics = dict(
                        value_loss=critic_loss,
                        critic_update_loss=critic_update_loss,
                        loss=loss,
                        aux_loss=aux_loss,
                        q=value.mean(),
                        reward_mean=minibatch.reward.mean(),
                        target_values=target_values.mean(),
                        q_min=value.min(),
                        q_max=value.max(),
                        target_values_min=target_values.min(),
                        target_values_max=target_values.max(),
                        reward_min=minibatch.reward.min(),
                        reward_max=minibatch.reward.max(),
                    )
                    
                    if cfg.log_pnorm:
                        critic_pnorm = utils.tree_norm(params)
                        metrics["critic_pnorm"] = critic_pnorm

                    return loss, metrics

                critic_grad_fn = jax.value_and_grad(critic_loss_fn, has_aux=True)
                output, critic_grads = critic_grad_fn(train_state.critic.params)
                critic_train_state = train_state.critic.apply_gradients(critic_grads)
                train_state = train_state.replace(
                    critic=critic_train_state,
                )
                critic_metrics = output[1]
                # log critic gradient norm
                if cfg.log_gnorm:
                    critic_gnorm = utils.tree_norm(critic_grads)
                    critic_metrics["critic_gnorm"] = critic_gnorm

                return (idx + 1, train_state), {
                    **critic_metrics,
                }

            # Shuffle data and split into mini-batches
            key, shuffle_key = jax.random.split(key)
            mini_batch_size = (cfg.num_steps * cfg.num_envs) // cfg.num_mini_batches
            indices = jax.random.permutation(shuffle_key, cfg.num_steps * cfg.num_envs)
            minibatch_idxs = jax.tree.map(
                lambda x: x.reshape(
                    (cfg.num_mini_batches, mini_batch_size, *x.shape[1:])
                ),
                indices,
            )

            # Run model update for each mini-batch
            train_state, metrics = jax.lax.scan(
                minibatch_update, train_state, minibatch_idxs
            )
            # Compute mean metrics across mini-batches
            metrics = jax.tree.map(lambda x: x.mean(0), metrics)
            return train_state, metrics

        def update_actor(train_state, key) -> tuple[SACTrainState, dict[str, jax.Array]]:
            def minibatch_update(carry, indices_and_key):
                indices, key = indices_and_key
                idx, train_state = carry
                # Sample data at indices from the batch
                minibatch, target_values = jax.tree.map(
                    lambda x: jnp.take(x, indices, axis=0), data
                )

                def adjoint_matching(params):
                    # 1. Merge models
                    actor_model = nnx.merge(train_state.actor.graphdef, params)

                    # Access the diffusion components
                    diffusion = actor_model.diffusion_model
                    old_diffusion = actor_target_model.diffusion_model
                    scheduler = diffusion.noise_scheduler

                    batch_size = minibatch.action.shape[0] * cfg.batch_repetitions

                    a_T_unsquashed = jnp.repeat(minibatch.action_unsquashed, cfg.batch_repetitions, axis=0)
                    squashing_correction_grad = jnp.repeat(minibatch.squashing_correction_grad, cfg.batch_repetitions, axis=0)
                    obs = jnp.repeat(minibatch.obs, cfg.batch_repetitions, axis=0)
                    Q_value = jnp.repeat(minibatch.Q_value, cfg.batch_repetitions, axis=0)
                    Q_score = jnp.repeat(minibatch.Q_score, cfg.batch_repetitions, axis=0)
                    log_weights = jnp.repeat(minibatch.log_weights, cfg.batch_repetitions, axis=0)


                    # 1. Randomly sample time t between [0,1]
                    key_t, key_noise, key_kl, key_ent, key_sample = jax.random.split(key, 5)  # Split the key passed to actor_loss
                    t = jax.random.uniform(key_t, (batch_size, 1))

                    # 2. Randomly sample noise from N(0, I)
                    noise = jax.random.normal(key_noise, a_T_unsquashed.shape)

                    # 3. Sample a_t ~ Pi^0_{t|1}(.|X_1 = a_T): the pinned-Brownian reference
                    # bridge conditioned on the terminal latent sample a_T from the buffer
                    # (reciprocal projection), Gaussian with the scheduler's coefficients.

                    # Retrieve coefficients from scheduler (handling broadcasting)
                    mu_scale = scheduler.mu_t_0T_scale(t)  # Shape: (Batch, 1)
                    sigma_scale = scheduler.sigma_t_0T(t)  # Shape: (Batch, 1)
                    sigma_t = scheduler.sigma_t(t)  # Shape: (Batch, 1)

                    # a_t = mu(t) * a_T + sigma(t) * noise
                    a_t = mu_scale * a_T_unsquashed + sigma_scale * noise

                    # 4. Evaluate the control u = sigma_t * fwd_model(a_t, obs, t) for the
                    # current actor and the old actor (trust-region anchor).
                    ctrl = sigma_t * jax.vmap(diffusion.fwd_model, in_axes=(0, 0, 0))(a_t, obs, t)
                    old_ctrl = sigma_t * jax.vmap(old_diffusion.fwd_model, in_axes=(0, 0, 0))(a_t, obs, t)

                    # 5. Adjoint matching: regress u onto ctrl_target = -sigma_t * adjoint_state.
                    # The reference drift vanishes, so the lean adjoint is constant in t and
                    # equals the terminal gradient
                    #   adjoint_state = nabla_x [ log Pi^0_1(x) - log|det J_f(x)| - Q(f(x), s)/eta ]
                    # at x = a_T, i.e. ctrl_target = sigma_t * nabla_x log(pi~*(x)/Pi^0_1(x))
                    # with pi~* \propto exp(Q/eta) |det J_f|: the control follows the terminal
                    # tilt of the reference toward the squashed-space target (paper eq. 13).
                    # The full squashing term is kept so any bijective squashing works; for
                    # the scaled-erf choice it cancels analytically against nabla log Pi^0_1
                    # (paper eq. 15).
                    temperature = actor_model.temperature()
                    temp_scaler = jax.lax.stop_gradient(temperature)
                    nabla_p_T_ref = -a_T_unsquashed / scheduler.sigma_T_0() ** 2  # nabla_x log Pi^0_1, Pi^0_1 = N(0, sigma_T_0^2 I)
                    adjoint_state = (nabla_p_T_ref - squashing_correction_grad) - (Q_score / temp_scaler)
                    ctrl_target = - sigma_t * adjoint_state

                    adjoint_loss = 0.5 * jnp.sum(jnp.square(ctrl - ctrl_target), axis=-1)

                    unscaled_adjoint_loss = adjoint_loss

                    # 1/sigma_t loss weighting
                    w_t = sigma_t.squeeze() ** -1
                    # Compute importance weights
                    log_importance_weights = log_weights.squeeze(-1) + Q_value / jax.lax.stop_gradient(temperature)

                    adjoint_loss = adjoint_loss * w_t

                    match (cfg.trust_region_lagrangian, cfg.trust_region_time_weighting, cfg.trust_region_granularity):
                        case ("dual_descent", _, _):
                            lagrangian = actor_model.lagrangian()
                            temperature = actor_model.temperature()
                        case ("dual_optimal_geometric_average", True, "avg"):
                            opt_lm = find_optimum_kl_lagrangian(w_t, old_ctrl, ctrl_target, eps=cfg.kl_bound)
                            dual_lm = actor_model.lagrangian()
                            lagrangian = dual_lm * jax.lax.stop_gradient(opt_lm)
                            temperature = actor_model.temperature()
                        case _:
                            raise NotImplementedError(_)

                    kl_scale = jax.lax.stop_gradient(lagrangian)
                    kl_loss = jnp.mean(0.5 * jnp.mean(jnp.sum(jnp.square(ctrl - old_ctrl), axis=-1)))

                    if cfg.reduce_kl:
                        actor_loss = adjoint_loss + kl_scale * kl_loss
                    else:
                        actor_loss = adjoint_loss

                    # # SAC target entropy loss
                    entropy = minibatch.log_weights.mean()


                    optimal_entropy = compute_entropy_via_importance_sampling(log_weights.reshape(-1),
                                                                              Q_value.reshape(-1),
                                                                              temp_scaler)

                    target_entropy = action_size_target + entropy
                    target_entropy_loss = temperature * jax.lax.stop_gradient(target_entropy)
                    target_entropy_loss = target_entropy_loss.mean()
                    #
                    # # Lagrangian constraint (follows temperature update)
                    lagrangian_loss = -lagrangian * jax.lax.stop_gradient(kl_loss - cfg.kl_bound)
                    lagrangian_loss = lagrangian_loss.mean()
                    #
                    # total loss
                    loss = jnp.mean(actor_loss)
                    if cfg.update_entropy_lagrangian:
                        loss += target_entropy_loss

                    if cfg.update_kl_lagrangian:
                        loss += lagrangian_loss

                    # log diffusion coefficient (detached for safe logging)
                    ctrl_norm = 0.5 * jnp.mean(jnp.sum(jnp.square(ctrl), axis=-1))
                    old_ctrl_norm = 0.5 * jnp.mean(jnp.sum(jnp.square(old_ctrl), axis=-1))
                    nabla_Q_norm = jnp.mean(jnp.sum(jnp.square(minibatch.Q_score), axis=-1))
                    squashing_correction_grad_norm = jnp.mean(jnp.sum(jnp.square(minibatch.squashing_correction_grad), axis=-1))
                    nabla_p_T_ref_grad_norm = jnp.mean(jnp.sum(jnp.square(-a_T_unsquashed / scheduler.sigma_T_0() ** 2), axis=-1))
                    adjoint_norm = jnp.mean(jnp.sum(jnp.square(sigma_t * adjoint_state), axis=-1))
                    weighted_adjoint_norm = jnp.mean(w_t * jnp.sum(jnp.square(ctrl_target), axis=-1))
                    mean_Q_vals = jnp.mean(Q_value)
                    std_Q_vals = jnp.std(Q_value)
                    mean_loss_weight = jnp.mean(w_t)


                    max_ctrl_target = jnp.max(ctrl_target)
                    max_unscaled_adjoint_loss = jnp.max(unscaled_adjoint_loss)
                    max_nabla_Q_normsq = jnp.max(jnp.sum(jnp.square(minibatch.Q_score), axis=-1))
                    max_scaled_nabla_Q_normsq = jnp.max(jnp.sum(jnp.square(minibatch.Q_score / temp_scaler), axis=-1))
                    max_Q_val = jnp.max(Q_value)
                    max_ctrl_normsq = jnp.max(jnp.sum(jnp.square(ctrl), axis=-1))
                    max_old_ctrl_normsq = jnp.max(jnp.sum(jnp.square(old_ctrl), axis=-1))
                    max_unsqaushed_action = jnp.max(jnp.abs(minibatch.action_unsquashed))
                    max_kl_loss = jnp.max( 0.5 * jnp.sum(jnp.square(ctrl - old_ctrl), axis=-1) )
                    max_log_path_weight_deterministic = jnp.max(minibatch.log_path_weight_deterministic)
                    max_log_path_weight_stochastic = jnp.max(minibatch.log_path_weight_stochastic)
                    max_log_p_T_ref_weight = jnp.max(minibatch.log_p_T_ref)
                    max_log_weights = jnp.max(log_weights)
                    max_squashing_correction_val = jnp.max(minibatch.squashing_correction_val)
                    min_log_path_weight_deterministic = jnp.min(minibatch.log_path_weight_deterministic)
                    min_log_path_weight_stochastic = jnp.min(minibatch.log_path_weight_stochastic)
                    min_log_p_T_ref_weight = jnp.min(minibatch.log_p_T_ref)
                    min_log_weights = jnp.min(log_weights)
                    min_squashing_correction_val = jnp.min(minibatch.squashing_correction_val)


                    metrics = dict(
                        mean_loss_weight=mean_loss_weight,
                        actor_loss=actor_loss,
                        loss=loss,
                        temp=actor_model.temperature(),
                        abs_batch_action=jnp.abs(minibatch.action).mean(),
                        abs_batch_action_unsquashed=jnp.abs(minibatch.action_unsquashed).mean(),
                        action_norm_diff=jnp.mean(jnp.sum(jnp.square(minibatch.action - minibatch.action_unsquashed), axis=-1)),
                        action_norm=jnp.mean(jnp.sum(jnp.square(minibatch.action), axis=-1)),
                        action_norm_unsquashed=jnp.mean(jnp.sum(jnp.square(minibatch.action_unsquashed), axis=-1)),
                        adjoint_norm=adjoint_norm,
                        weighted_adjoint_norm=weighted_adjoint_norm,
                        mean_Q_vals=mean_Q_vals,
                        std_Q_vals=std_Q_vals,
                        squashing_correction_grad_norm=squashing_correction_grad_norm,
                        nabla_p_T_ref_grad_norm=nabla_p_T_ref_grad_norm,
                        reward_mean=minibatch.reward.mean(),
                        soft_reward_mean=minibatch.soft_reward.mean(),
                        kl=kl_loss,
                        kl_loss=kl_loss,
                        scaled_kl_loss=kl_scale * kl_loss,
                        adjoint_loss=adjoint_loss,
                        loss_ratio=adjoint_loss / (kl_scale * kl_loss),
                        ctrl_norm=ctrl_norm,
                        old_ctrl_norm=old_ctrl_norm,
                        nabla_Q_norm=nabla_Q_norm,
                        ESS=compute_reverse_ess(log_importance_weights),
                        m_step_lagrangian_loss=lagrangian_loss,
                        m_step_lagrangian=lagrangian,
                        entropy=jnp.mean(minibatch.log_weights),
                        optimal_entropy=optimal_entropy,
                        entropy_loss=target_entropy_loss,
                        entropy_temp=actor_model.temperature(),
                        target_entropy=-action_size_target,
                        log_path_weight_deterministic=minibatch.log_path_weight_deterministic.mean(),
                        log_path_weight_stochastic=minibatch.log_path_weight_stochastic.mean(),
                        log_p_T_ref_weight=minibatch.log_p_T_ref.mean(),
                        log_weights=log_weights.mean(),
                        squashing_correction_val=minibatch.squashing_correction_val.mean(),
                        max_ctrl_target = max_ctrl_target,
                        max_unscaled_adjoint_loss = max_unscaled_adjoint_loss,
                        max_nabla_Q_normsq = max_nabla_Q_normsq,
                        max_scaled_nabla_Q_normsq = max_scaled_nabla_Q_normsq,
                        max_Q_val = max_Q_val,
                        max_ctrl_normsq = max_ctrl_normsq,
                        max_old_ctrl_normsq = max_old_ctrl_normsq,
                        max_unsqaushed_action = max_unsqaushed_action,
                        max_kl_loss = max_kl_loss,
                        max_log_path_weight_deterministic = max_log_path_weight_deterministic,
                        max_log_path_weight_stochastic = max_log_path_weight_stochastic,
                        max_log_p_T_ref_weight = max_log_p_T_ref_weight,
                        max_log_weights = max_log_weights,
                        max_squashing_correction_val = max_squashing_correction_val,
                        min_log_path_weight_deterministic = min_log_path_weight_deterministic,
                        min_log_path_weight_stochastic = min_log_path_weight_stochastic,
                        min_log_p_T_ref_weight = min_log_p_T_ref_weight,
                        min_log_weights = min_log_weights,
                        min_squashing_correction_val = min_squashing_correction_val,
                    )
                    if cfg.trust_region_lagrangian == "dual_optimal_geometric_average":
                        metrics["opt_lm"] = opt_lm.mean()
                        metrics["dual_lm"] = dual_lm.mean()

                    if cfg.log_pnorm:
                        actor_pnorm = utils.tree_norm(params)
                        metrics["actor_pnorm"] = actor_pnorm

                    return loss, metrics

                actor_loss = adjoint_matching
                actor_grad_fn = jax.value_and_grad(actor_loss, has_aux=True)
                output, actor_grads = actor_grad_fn(train_state.actor.params)
                actor_train_state = train_state.actor.apply_gradients(actor_grads)
                actor_metrics = output[1]
                train_state = train_state.replace(
                    actor=actor_train_state,
                )
                # log actor gradient norm
                if cfg.log_gnorm:
                    actor_gnorm = utils.tree_norm(actor_grads)
                    actor_metrics["actor_gnorm"] = actor_gnorm
                return (idx + 1, train_state), {
                    **actor_metrics,
                }

            # Shuffle data and split into mini-batches
            key, shuffle_key = jax.random.split(key)
            mini_batch_size = (cfg.num_steps * cfg.num_envs) // cfg.num_mini_batches
            indices = jax.random.permutation(shuffle_key, cfg.num_steps * cfg.num_envs)
            minibatch_idxs = jax.tree.map(
                lambda x: x.reshape(
                    (cfg.num_mini_batches, mini_batch_size, *x.shape[1:])
                ),
                indices,
            )

            # Run model update for each mini-batch
            train_state, metrics = jax.lax.scan(
                minibatch_update, train_state, (minibatch_idxs, jax.random.split(key, cfg.num_mini_batches))
            )
            # Compute mean metrics across mini-batches
            metrics = jax.tree.map(lambda x: x.mean(0), metrics)
            return train_state, metrics

        def update_Q(train_state: SACTrainState, data: Transition) -> Transition:
            """
            Updates importance weights to: Q(s,a) - log_pi_behavior(a|s)
            This results in exp(weight) = exp(Q) / pi_behavior
            """
            critic_params = train_state.critic_target.params if cfg.use_target_critic_for_actor else train_state.critic.params
            critic_model = nnx.merge(train_state.critic.graphdef, critic_params)
            actor_model = nnx.merge(train_state.actor.graphdef, train_state.actor.params)


            def Q_score_clipping(grads, max_norm=1., percentile=95.0):
                # 1. Compute norms per sample
                sample_norms = jnp.linalg.norm(grads, axis=-1, keepdims=True)

                # 2. Determine the robust ceiling for this batch
                # We clip gradients that are outliers relative to the CURRENT batch
                batch_percentile = jnp.percentile(sample_norms, percentile)

                # The threshold is the tighter of the global config max_norm OR the batch's p95
                # This prevents the threshold from being too high if the whole batch is explosive
                clip_threshold = jnp.minimum(batch_percentile, max_norm)

                # 3. Compute scaling factors
                # Prevent division by zero with 1e-6
                scale_factor = jnp.where(
                    sample_norms > clip_threshold,
                    clip_threshold / (sample_norms + 1e-6),
                    1.0
                )

                return grads * scale_factor, jnp.mean(scale_factor < 1.0)


            if cfg.Q_score_max_norm_for_squashed:
                def squashed_critic(obs, action_squashed):
                    return critic_model.critic(obs, action_squashed)

                def squash_action(action_unsquashed):
                    return actor_model.squashing.forward(action_unsquashed)

                # setup batched critic gradient in squashed space
                dq_dsa_fn = jax.value_and_grad(squashed_critic, argnums=1)
                batched_dq_dsa_fn = jax.vmap(dq_dsa_fn, in_axes=(0, 0))

                # setup batched squashing gradient
                batched_squash = jax.vmap(squash_action, in_axes=(0))
                squashed_action, dsa_da_fn = jax.vjp(batched_squash, data[0].action_unsquashed)

                # compute squashed actions gradients
                squashed_Q_value, squashed_Q_score = batched_dq_dsa_fn(data[0].critic_obs, squashed_action)
                Q_value, squashed_Q_score = jax.lax.stop_gradient(squashed_Q_value), jax.lax.stop_gradient(squashed_Q_score)
                # clip them
                squashed_Q_score_clipped, pct_clipped = Q_score_clipping(squashed_Q_score, cfg.Q_score_max_norm, cfg.Q_score_max_percentile)
                # then unsquash them
                (Q_score_clipped,) = dsa_da_fn(squashed_Q_score_clipped)
            else:
                def single_sample_critic(obs, action_unsquashed):
                    action = actor_model.squashing.forward(action_unsquashed)
                    q_val = critic_model.critic(obs, action)
                    return jnp.squeeze(q_val)

                # Compute gradients w.r.t action (argnums=1)
                dq_da_fn = jax.value_and_grad(single_sample_critic, argnums=1)
                batched_dq_da_fn = jax.vmap(dq_da_fn, in_axes=(0, 0))

                Q_value, Q_score = batched_dq_da_fn(data[0].critic_obs, data[0].action_unsquashed)
                Q_value, Q_score = jax.lax.stop_gradient(Q_value), jax.lax.stop_gradient(Q_score)

                Q_score_clipped, pct_clipped = Q_score_clipping(Q_score, cfg.Q_score_max_norm, cfg.Q_score_max_percentile)

            return data[0].replace(Q_value=Q_value, Q_score=Q_score_clipped)

        # Update the model for a number of epochs
        key, train_key = jax.random.split(key)
        (_, train_state), update_metrics_critic = jax.lax.scan(
            f=update_critic,
            init=(1, train_state),
            xs=jax.random.split(train_key, cfg.num_epochs_critic),
        )

        # Polyak averaging of critic parameters
        new_target_params = optax.incremental_update(
            train_state.critic.params,
            train_state.critic_target.params,
            step_size=cfg.polyak
        )
        # Already update the target critic, so we use the new one in update_Q
        train_state = train_state.replace(
            critic_target=train_state.critic_target.replace(params=new_target_params)
        )

        new_data = update_Q(train_state, data)
        data = (new_data, data[1])

        key, train_key = jax.random.split(key)
        (_, train_state), update_metrics_actor = jax.lax.scan(
            f=update_actor,
            init=(1, train_state),
            xs=jax.random.split(train_key, cfg.num_epochs_actor),
        )



        update_metrics = {**update_metrics_actor, **update_metrics_critic}
        # Get metrics from the last epoch
        update_metrics = jax.tree.map(lambda x: x[-1], update_metrics)

        return train_state, update_metrics

    eval_interval = int(
        (cfg.total_time_steps / (cfg.num_steps * cfg.num_envs)) // cfg.num_eval
    )

    def split_iteration_keys(key: PRNGKeyArray) -> tuple[PRNGKeyArray, PRNGKeyArray]:
        """Per-seed key derivation of one training iteration (train keys, eval seed key)."""
        train_key, eval_key = jax.random.split(key)
        eval_key, init_seed_key = jax.random.split(eval_key)
        return train_key, init_seed_key

    def train_interval(
        key: PRNGKeyArray, train_state: SACTrainState
    ) -> tuple[SACTrainState, dict[str, Float[Array, ""]], dict[str, Float[Array, ""]]]:
        """One eval interval of training: `eval_interval` rollout+learn steps."""

        def train_step(
            state: SACTrainState, key: PRNGKeyArray
        ) -> tuple[SACTrainState, dict[str, Float[Array, ""]]]:
            key, rollout_key, learn_key = jax.random.split(key, 3)
            transitions, state = collect_rollout(key=rollout_key, train_state=state)
            state, update_metrics = learn_step(
                key=learn_key, train_state=state, batch=transitions
            )
            metrics = {**update_metrics}
            state = state.replace(iteration=state.iteration + 1)
            return state, metrics

        train_state, train_metrics = jax.lax.scan(
            f=train_step,
            init=train_state,
            xs=jax.random.split(key, eval_interval),
        )
        train_metrics = jax.tree.map(lambda x: x[-1], train_metrics)

        train_returns = {
            "train/episode_return": train_state.last_env_state.info[
                "returned_episode_returns"
            ].mean(),
            "train/episode_length": train_state.last_env_state.info[
                "returned_episode_lengths"
            ].mean(),
        }
        return train_state, train_metrics, train_returns

    def host_train_interval(
        keys: PRNGKeyArray, train_state: SACTrainState
    ) -> tuple[
        SACTrainState, dict[str, Float[Array, "seeds"]], dict[str, Float[Array, "seeds"]]
    ]:
        """One eval interval for a host-stepped env: a Python loop over rollout+learn.

        `keys` carries the seed axis; the key derivation mirrors the JAX-native
        `train_interval` per seed.
        """
        train_metrics = None
        step_keys = jit_split_interval_keys(keys)  # (seeds, eval_interval, 2)
        for t in range(eval_interval):
            _, rollout_keys, learn_keys = jit_split_step_keys(
                step_keys[:, t]
            ).transpose(1, 0, 2)
            transitions, train_state = host_collect_rollout(
                keys=rollout_keys, train_state=train_state
            )
            train_state, train_metrics = jit_learn_step(
                key=learn_keys, train_state=train_state, batch=transitions
            )
            train_state = train_state.replace(iteration=train_state.iteration + 1)

        train_returns = {
            "train/episode_return": train_state.last_env_state.info[
                "returned_episode_returns"
            ].mean(axis=-1),
            "train/episode_length": train_state.last_env_state.info[
                "returned_episode_lengths"
            ].mean(axis=-1),
        }
        return train_state, train_metrics, train_returns

    init_fn, init_models, init_env, assemble = make_init(cfg, env, env_params)

    # Each phase is a separate jit: the eval graphs are compiled once instead of being
    # inlined into the training graph once per Best-of-N value, and the host loop can
    # log between iterations without a callback inside the graph. Every phase is vmapped
    # over the seed axis on both paths (size 1 for a single seed); a host-stepped env
    # additionally reshapes to its flat physical batch at the env boundary.
    jit_split_keys = jax.jit(jax.vmap(split_iteration_keys))
    if jax_native:
        jit_init = jax.jit(jax.vmap(init_fn))
        run_train_interval = jax.jit(jax.vmap(train_interval))
        run_sde_eval = jax.jit(jax.vmap(sde_eval_fn))
        run_best_of_n_evals = {
            n: jax.jit(jax.vmap(fn)) for n, fn in best_of_n_eval_fns.items()
        }
    else:
        jit_init_models = jax.jit(jax.vmap(init_models))
        jit_learn_step = jax.jit(jax.vmap(learn_step))
        jit_split_interval_keys = jax.jit(
            jax.vmap(lambda k: jax.random.split(k, eval_interval))
        )
        jit_split_step_keys = jax.jit(jax.vmap(lambda k: jax.random.split(k, 3)))
        jit_wrap_reset = jax.jit(
            jax.vmap(
                partial(
                    wrap_reset_stats,
                    num_envs=cfg.num_envs,
                    normalize=cfg.normalize_env,
                )
            )
        )
        run_train_interval = host_train_interval
        run_sde_eval = sde_eval_fn
        run_best_of_n_evals = best_of_n_eval_fns

    def host_init(key: PRNGKeyArray) -> SACTrainState:
        """Init for a host-stepped env: the models are jitted and vmapped over the seed
        axis, `env.reset` runs eagerly on the flat physical batch."""
        keys, *trainstates = jit_init_models(jax.random.split(key, num_seeds))

        obs, critic_obs, raw_state = env.reset(None)
        obs, critic_obs, env_state = jit_wrap_reset(
            *unflatten_seeds((obs, critic_obs, raw_state))
        )

        if cfg.use_staggered_resets:
            # Mirror `init_env`: deterministic group stagger, identical for every seed
            # (each seed's env block gets the same group layout). Runs eagerly because
            # setting the state may sync a torch engine (ManiSkill), which cannot
            # happen under a trace. No effect for humanoid_bench, whose subprocess
            # TimeLimit cannot be pre-aged (same as the torch trainer).
            offsets = utils.staggered_reset_offsets(
                cfg.num_envs,
                cfg.max_episode_steps,
                cfg.num_steps,
                cfg.num_stagger_blocks,
            )
            _env_state = env_state.unwrapped()
            _env_state.info["steps"] = jnp.broadcast_to(
                offsets, (num_seeds, cfg.num_envs)
            )
            env_state = env_state.set_env_state(_env_state)
        else:
            # Mirror `init_env`: randomize the initial time steps per seed so the envs do
            # not step in tandem. Runs eagerly because setting the state may sync a torch
            # engine (ManiSkill), which cannot happen under a trace.
            def randomize_key(k):
                k, _env_key = jax.random.split(k)
                _k, randomize_steps_key = jax.random.split(k)
                return randomize_steps_key

            _env_state = env_state.unwrapped()
            _env_state.info["steps"] = jax.vmap(
                lambda k, steps: jax.random.randint(
                    k, steps.shape, 0, cfg.max_episode_steps
                ).astype(jnp.float32)
            )(jax.vmap(randomize_key)(keys), _env_state.info["steps"])
            env_state = env_state.set_env_state(_env_state)

        state = assemble(trainstates, obs, critic_obs, env_state)
        # `assemble` writes scalar bookkeeping; give it the seed axis the vmapped
        # phases expect (the JAX-native path gets this from vmapping `init_fn`).
        return state.replace(
            iteration=jnp.zeros(num_seeds, dtype=jnp.int32),
            time_steps=jnp.zeros(num_seeds, dtype=jnp.int32),
        )

    def train_fn(
        key: PRNGKeyArray, cfg: ReppoConfig
    ) -> tuple[SACTrainState, dict[str, Float[Array, "iterations seeds"]]]:
        num_train_steps = cfg.total_time_steps // (cfg.num_steps * cfg.num_envs)
        num_iterations = num_train_steps // eval_interval + int(
            num_train_steps % eval_interval != 0
        )
        key, init_key = jax.random.split(key)
        if jax_native:
            train_state = jit_init(jax.random.split(init_key, num_seeds))
        else:
            train_state = host_init(init_key)

        keys = jax.random.split(key, num_iterations)
        metric_history = []
        for iteration_key in keys:
            _, subkey = jax.random.split(iteration_key)
            train_keys, init_seed_keys = jit_split_keys(
                jax.random.split(subkey, num_seeds)
            )

            eval_time_steps = train_state.time_steps
            norm_state = train_state.last_env_state if cfg.normalize_env else None

            # The same init seed is used for the SDE and all Best-of-N evaluations.
            eval_metrics = run_sde_eval(init_seed_keys, train_state, norm_state)
            for best_of_n, run_eval in run_best_of_n_evals.items():
                eval_metrics_bon = run_eval(init_seed_keys, train_state, norm_state)
                bon_suffix = f"bon_{best_of_n:03d}"
                eval_metrics.update(
                    {f"{k}_{bon_suffix}": v for k, v in eval_metrics_bon.items()}
                )

            train_state, train_metrics, train_returns = run_train_interval(
                train_keys, train_state
            )

            metrics = {
                "time_step": train_state.time_steps,
                "eval/time_step": eval_time_steps,
                **utils.prefix_dict("train", train_metrics),
                **utils.prefix_dict("eval", eval_metrics),
                **train_returns,
            }
            # log_callback adds bookkeeping keys, so it must not see the stacked dict.
            log_callback(train_state, dict(metrics))
            metric_history.append(metrics)

        metrics = jax.tree.map(lambda *xs: jnp.stack(xs), *metric_history)
        return train_state, metrics

    return train_fn

# type object
def run(cfg: DictConfig, trial: optuna.Trial | None) -> float:
    """
    Run a single trial of the SAC training process with hyperparameter tuning.
    Args:
        cfg (DictConfig): Configuration for the SAC training.
        trial (optuna.Trial | None): Optuna trial object for hyperparameter tuning.
    Returns:
        float: The mean episode return from the trial.
    """
    utils.setup_compilation_cache()
    sweep_metrics = []

    if trial is not None:
        # Set hyperparameters from the trial
        for name, values in cfg.trial_spec.items():
            if name in cfg.hyperparameters:
                sampled_value = _get_optuna_type(trial, name, values)
                # optuna can return np.float64, which OmegaConf rejects
                if isinstance(sampled_value, np.float64):
                    sampled_value = float(sampled_value)
                cfg.hyperparameters[name] = sampled_value
            else:
                raise ValueError(f"Hyperparameter {name} not found in config.")

    try:
        with open("completed_trials.txt", "r") as f:
            completed_trials = int(f.read())
    except FileNotFoundError:
        completed_trials = 0

    metric_history = []

    def log_callback(state, metrics):
        metrics["sys_time"] = time.perf_counter()
        if len(metric_history) > 0:
            num_env_steps = state.time_steps[0] - metric_history[-1]["time_step"][0]
            seconds = metrics["sys_time"] - metric_history[-1]["sys_time"]
            sps = num_env_steps / seconds
        else:
            sps = 0

        metric_history.append(metrics)
        time_step = int(state.time_steps[0])
        episode_return = metrics["eval/episode_return"].mean()
        eval_length = metrics["eval/episode_length"].mean()
        eval_success = metrics["eval/episode_success"].mean()

        log_msg = f"step={time_step} episode_return={episode_return:.3f}, episode_length={eval_length:.3f}, episode_success={eval_success:.3f}"
        
        # Log Best-of-N metrics if available
        bon_metrics = {}
        for key in metrics.keys():
            if "episode_return_bon_" in key:
                # Extract N from key (e.g., "eval/episode_return_bon_016" -> 16)
                best_of_n = int(key.split("_bon_")[-1])
                bon_return = metrics[key].mean()
                bon_metrics[f"bon_{best_of_n}"] = bon_return
                log_msg += f", bon_{best_of_n}_return={bon_return:.3f}"
        
        log_msg += f" sps={sps:.2f}"
        logging.info(log_msg)
        
        log_data = {
            "eval/episode_return": episode_return,
            "eval/episode_length": eval_length,
            # performance metric: steps per second
            "sps": sps,
            **jax.tree.map(jnp.mean, utils.filter_prefix("train", metrics)),
        }
        
        # Add all eval metrics (SDE and Best-of-N variants)
        for key, value in metrics.items():
            if key.startswith("eval/"):
                log_data[key] = value.mean() if hasattr(value, 'mean') else value

        # With multiple seeds the canonical keys above hold the seed mean; also emit each
        # seed under `<key>/seed_i` so per-seed curves survive. The leading axis of every
        # eval/train array is the seed axis (size 1 for a single seed, so this is skipped).
        num_seeds = int(jnp.asarray(metrics["eval/episode_return"]).shape[0])
        if num_seeds > 1:
            for key, value in metrics.items():
                if not (key.startswith("eval/") or key.startswith("train/")):
                    continue
                per_seed = jnp.asarray(value).reshape(num_seeds, -1).mean(axis=1)
                for i in range(num_seeds):
                    log_data[f"{key}/seed_{i}"] = per_seed[i]

        wandb.log(log_data, step=time_step)

    # Set up the experiment
    if cfg.env.type == "mjx":
        env = MjxGymnaxWrapper(
            cfg.env.name,
            episode_length=cfg.env.max_episode_steps,
            push_distractions=cfg.env.get("push_distractions", False),
            asymmetric_observation=cfg.env.get("asymmetric_observation", False),
        )
        eval_env = MjxGymnaxWrapper(
            cfg.env.name,
            episode_length=cfg.env.max_episode_steps,
            push_distractions=cfg.env.get("push_distractions", False),
            asymmetric_observation=cfg.env.get("asymmetric_observation", False),
        )
    elif cfg.env.type == "humanoid_bench":
        # One physical vec env batches every seed; the trainer reshapes at the boundary.
        num_physical_envs = cfg.num_seeds * cfg.hyperparameters.num_envs
        env = HumanoidBenchGymnaxWrapper(
            cfg.env.name,
            num_envs=num_physical_envs,
            num_subproc=cfg.hyperparameters.get("num_subproc", 1),
        )
        eval_env = HumanoidBenchGymnaxWrapper(
            cfg.env.name,
            num_envs=num_physical_envs,
            num_subproc=cfg.hyperparameters.get("num_subproc", 1),
        )
    elif cfg.env.type == "maniskill":
        num_physical_envs = cfg.num_seeds * cfg.hyperparameters.num_envs
        env = ManiSkillGymnaxWrapper(
            env_name=cfg.env.name,
            num_envs=num_physical_envs,
            reconfiguration_freq=None, # Training usually None
            partial_reset=cfg.env.partial_reset,
            asymmetric_observation=False,
            env_kwargs=cfg.env.env_kwargs,
            has_final_obs=cfg.env.get("has_final_obs", False),
            # Torch parity: training ends episodes on termination under partial_reset.
            ignore_terminations=not cfg.env.partial_reset,
        )
        eval_env = ManiSkillGymnaxWrapper(
            env_name=cfg.env.name,
            num_envs=num_physical_envs,
            reconfiguration_freq=1,
            partial_reset=cfg.env.partial_reset,
            asymmetric_observation=False,
            env_kwargs=cfg.env.env_kwargs,
            has_final_obs=cfg.env.get("has_final_obs", False),
            ignore_terminations=True,
        )
    else:
        raise ValueError(f"Unknown environment type: {cfg.env.type}")

    # JAX-native envs step inside the traced rollout; the others are stepped by a host
    # loop between jitted phases, which keeps them out of the graph entirely.
    jax_native = cfg.env.get("jax_native", cfg.env.type == "mjx")
    if cfg.env.type == "mjx" and not jax_native:
        raise ValueError("mjx environments are JAX-native; set env.jax_native: true")

    if (
        cfg.hyperparameters.get("use_staggered_resets", False)
        and cfg.env.type == "humanoid_bench"
    ):
        logging.warning(
            "use_staggered_resets has no effect for humanoid_bench: the subprocess "
            "TimeLimit cannot be pre-aged (same limitation as the torch trainer)."
        )

    train_fn = make_train_fn(
        cfg=ReppoConfig(**cfg.hyperparameters),
        env=env,
        eval_env=eval_env,
        log_callback=log_callback,
        num_seeds=cfg.num_seeds,
        jax_native=jax_native,
        partial_reset=cfg.env.get("partial_reset", False),
    )

    for i in range(completed_trials, cfg.num_trials):
        cfg.seed = cfg.seed + i

        wandb.init(
            mode=cfg.wandb.mode,
            project=cfg.wandb.project,
            entity=cfg.wandb.entity,
            group=cfg.wandb.group,
            tags=[
                cfg.name,
                cfg.env.name,
                cfg.env.type,
                "hp_tune" if trial is not None else "val",
                *cfg.tags,
            ],
            config=OmegaConf.to_container(cfg),
            name=f"{cfg.name}-{cfg.env.name.lower()}",
            save_code=True,
        )

        logging.info(OmegaConf.to_yaml(cfg))

        key = jax.random.PRNGKey(cfg.seed)
        start = time.perf_counter()
        _, metrics = train_fn(key, ReppoConfig(**cfg.hyperparameters))
        jax.block_until_ready(metrics)
        duration = time.perf_counter() - start

        # Save metrics and finish the run
        logging.info(f"Training took {duration:.2f} seconds.")
        jnp.savez("metrics.npz", **metrics)
        wandb.finish()

        sweep_metrics.append(metrics["eval/episode_return"])

        with open("completed_trials.txt", "w") as f:
            f.write(str(i))

    sweep_metrics_array = jnp.array(sweep_metrics)
    return (0.1 * sweep_metrics_array.mean() + sweep_metrics_array[:, -1].mean()).item()


@hydra.main(version_base=None, config_path="../../config", config_name="amdp_reppo")
def main(cfg: DictConfig):
    try:
        run(cfg, trial=None)
    except Exception as ex:
        traceback.print_tb(ex.__traceback__)
        traceback.print_exception(ex)


if __name__ == "__main__":
    main()
