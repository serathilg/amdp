import logging
import time
import traceback
from typing import Callable

import hydra
import jax
import numpy as np
import optax
import optuna
from flax import nnx
from gymnax.environments.environment import Environment, EnvParams
from jax import numpy as jnp
from jaxtyping import PRNGKeyArray
from omegaconf import DictConfig, OmegaConf

import wandb
from src.env_utils.jax_wrappers import (
    ClipAction,
    LogWrapper,
    MjxGymnaxWrapper,
    NormalizeVec,
)
from src.jaxrl import utils
from src.jaxrl.lagrangian_utils import find_optimum_kl_lagrangian
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

logging.basicConfig(level=logging.INFO)


class ReppoConfig(ReppoConfigBase):
    best_of_n_values: list = None


def make_init(
    cfg: ReppoConfig,
    env: Environment,
    env_params: EnvParams = None,
) -> Callable[[jax.Array], SACTrainState]:
    def init(key: PRNGKeyArray) -> SACTrainState:
        # Number of calls to train_step
        key, model_key = jax.random.split(key)
        obs_dim=env.observation_space(env_params)[0].shape[0]
        critic_obs_dim=env.observation_space(env_params)[1].shape[0]
        action_dim=env.action_space(env_params).shape[0]

        # Initialize scheduler and integrators
        noise_schedule = hydra.utils.call(cfg.diffusion.noise_schedule)
        sde_integrator = hydra.utils.get_method(cfg.diffusion.sde_integrator)

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

        return SACTrainState(
            actor=actor_trainstate,
            actor_target=actor_target_trainstate,
            critic=critic_trainstate,
            critic_target=critic_target_trainstate,
            iteration=0,
            time_steps=0,
            last_env_state=env_state,
            last_obs=obs,
            last_critic_obs=critic_obs,
        )

    return init


def make_train_fn(
    cfg: ReppoConfig,
    env: Environment,
    eval_env: Environment,
    env_params: EnvParams = None,
    log_callback: Callable[[SACTrainState, dict[str, jax.Array]], None] | None = None,
    num_seeds: int = 1,
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
    """
    env = LogWrapper(env, cfg.num_envs)
    eval_env = LogWrapper(eval_env, cfg.num_envs)
    env = ClipAction(env)
    eval_env = ClipAction(eval_env)
    env = NormalizeVec(env, enable=cfg.normalize_env)
    eval_env = NormalizeVec(eval_env, enable=cfg.normalize_env)

    action_size_target = jnp.prod(jnp.array(env.action_space(env_params).shape)) * cfg.ent_target_mult

    def collect_rollout(
        key: PRNGKeyArray, train_state: SACTrainState
    ) -> tuple[Transition, SACTrainState]:
        actor_model = nnx.merge(train_state.actor.graphdef, train_state.actor.params)
        critic_model = nnx.merge(train_state.critic_target.graphdef, train_state.critic_target.params)

        offset = (
            jnp.arange(cfg.num_envs - cfg.exploration_base_envs)[:, None]
            * (cfg.exploration_noise_max - cfg.exploration_noise_min)
            / (cfg.num_envs - cfg.exploration_base_envs)
        ) + cfg.exploration_noise_min
        offset = jnp.concatenate(
            [
                jnp.ones((cfg.exploration_base_envs, 1)) * cfg.exploration_noise_min,
                offset,
            ],
            axis=0,
        )

        def step_env(carry, _) -> tuple[tuple, Transition]:
            key, env_state, train_state, obs, critic_obs = carry
            key, act_key, step_key = jax.random.split(key, 3)
            step_key = jax.random.split(step_key, cfg.num_envs)

            # get policy action
            sample = actor_model.sde_sample(act_key, obs, stop_grad=True)

            next_obs, next_critic_obs, next_env_state, reward, done, info = env.step(
                step_key, env_state, sample.action
            )

            # compute importance weights
            action = jnp.clip(sample.action, -0.999, 0.999)
            importance_weight = jnp.zeros((cfg.num_envs,))

            # compute next state embedding and value
            key, next_act_key = jax.random.split(key)
            next_sample = actor_model.sde_sample(next_act_key, next_obs, stop_grad=True)
            next_action = jax.lax.stop_gradient(next_sample.action)
            next_log_prob = -next_sample.log_weight
            next_log_prob = next_log_prob.sum(-1)
            # compute next state embedding and value
            next_emb, _, _, value = critic_model.forward(next_critic_obs, next_action)
            soft_reward = reward - cfg.gamma * next_log_prob.squeeze() * actor_model.temperature()
            transition = Transition(
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
                truncated=next_env_state.truncated,
                info=info,
                importance_weight=importance_weight,
                log_weights=sample.log_weight,
                Q_value=jnp.zeros_like(value),
                Q_score=jnp.zeros_like(action),
                log_path_weight_deterministic=sample.log_path_weight_deterministic,
                log_path_weight_stochastic=sample.log_path_weight_stochastic,
                log_p_T_ref=sample.log_p_T_ref,
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
        train_state = train_state.replace(
            last_env_state=last_env_state,
            last_obs=last_obs,
            last_critic_obs=last_critic_obs,
            time_steps=train_state.time_steps + cfg.num_steps * cfg.num_envs,
        )

        return transitions, train_state

    def compute_targets(batch: Transition):
        # compute n-step lambda estimates
        target_values = utils.compute_lambda_targets(batch, cfg.gamma, cfg.lmbda)
        # Reshape data to (num_steps * num_envs, ...)
        data = (batch, target_values)
        data = jax.tree.map(
            lambda x: x.reshape((cfg.num_steps * cfg.num_envs, *x.shape[2:])), data
        )
        return data

    def train_critic(key: PRNGKeyArray, train_state: SACTrainState, data) -> tuple[SACTrainState, dict[str, jax.Array]]:
        train_state = train_state.replace(
            actor_target=train_state.actor_target.replace(
                params=train_state.actor.params
            ),
        )
        
        def update_critic(carry, key):
            idx, train_state = carry

            def minibatch_update(train_state_mb, indices):
                # Sample data at indices from the batch
                minibatch, target_values = jax.tree.map(
                    lambda x: jnp.take(x, indices, axis=0), data
                )

                def critic_loss_fn(params):
                    critic_model = nnx.merge(train_state_mb.critic.graphdef, params)
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
                    aux_loss = jnp.mean(
                        (1 - minibatch.done.reshape(-1, 1))
                        * aux_loss, axis=-1)

                    # compute l2 error for logging
                    critic_loss = optax.squared_error(
                        value,
                        target_values,
                    )
                    critic_loss = jnp.mean(critic_loss)
                    loss = jnp.mean(
                        (1.0 - minibatch.truncated)
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
                output, critic_grads = critic_grad_fn(train_state_mb.critic.params)
                critic_train_state = train_state_mb.critic.apply_gradients(critic_grads)
                train_state_mb = train_state_mb.replace(
                    critic=critic_train_state,
                )
                critic_metrics = output[1]
                # log critic gradient norm
                if cfg.log_gnorm:
                    critic_gnorm = utils.tree_norm(critic_grads)
                    critic_metrics["critic_gnorm"] = critic_gnorm

                return train_state_mb, {
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
            return (idx + 1, train_state), metrics

        (_, train_state), update_metrics_critic = jax.lax.scan(
            f=update_critic,
            init=(1, train_state),
            xs=jax.random.split(key, cfg.num_epochs_critic),
        )

        # Polyak averaging of critic parameters
        new_target_params = optax.incremental_update(
            train_state.critic.params,
            train_state.critic_target.params,
            step_size=cfg.polyak
        )
        train_state = train_state.replace(
            critic_target=train_state.critic_target.replace(params=new_target_params)
        )
        update_metrics_critic = jax.tree.map(lambda x: x[-1], update_metrics_critic)

        return train_state, update_metrics_critic

    def do_update_Q(train_state: SACTrainState, data: tuple) -> tuple:
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

        new_data0 = data[0].replace(Q_value=Q_value, Q_score=Q_score_clipped)
        return new_data0, data[1]

    def train_actor(key: PRNGKeyArray, train_state: SACTrainState, data: tuple) -> tuple[SACTrainState, dict[str, jax.Array]]:
        actor_target_model = nnx.merge(
            train_state.actor_target.graphdef, train_state.actor_target.params
        )

        def update_actor(carry, key):
            idx, train_state = carry

            def minibatch_update(train_state_mb, indices_and_key):
                indices, key = indices_and_key
                
                minibatch, target_values = jax.tree.map(
                    lambda x: jnp.take(x, indices, axis=0), data
                )

                def adjoint_matching(params):
                    actor_model = nnx.merge(train_state_mb.actor.graphdef, params)

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

                    key_t, key_noise, key_kl, key_ent, key_sample = jax.random.split(key, 5)  
                    t = jax.random.uniform(key_t, (batch_size, 1))
                    noise = jax.random.normal(key_noise, a_T_unsquashed.shape)

                    mu_scale = scheduler.mu_t_0T_scale(t) 
                    sigma_scale = scheduler.sigma_t_0T(t) 
                    sigma_t = scheduler.sigma_t(t) 

                    a_t = mu_scale * a_T_unsquashed + sigma_scale * noise

                    ctrl = sigma_t * jax.vmap(diffusion.fwd_model, in_axes=(0, 0, 0))(a_t, obs, t)
                    old_ctrl = sigma_t * jax.vmap(old_diffusion.fwd_model, in_axes=(0, 0, 0))(a_t, obs, t)

                    temperature = actor_model.temperature() 
                    temp_scaler = jax.lax.stop_gradient(temperature)
                    nabla_p_T_ref = -a_T_unsquashed / scheduler.sigma_T_0() ** 2    
                    adjoint_state = (nabla_p_T_ref - squashing_correction_grad) - (Q_score / temp_scaler)
                    ctrl_target = - sigma_t * adjoint_state

                    adjoint_loss = 0.5 * jnp.sum(jnp.square(ctrl - ctrl_target), axis=-1)
                    
                    unscaled_adjoint_loss = adjoint_loss

                    # 1/sigma_t loss weighting
                    w_t = sigma_t.squeeze() ** -1
                    
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

                    entropy = minibatch.log_weights.mean()


                    target_entropy = action_size_target + entropy
                    target_entropy_loss = temperature * jax.lax.stop_gradient(target_entropy)
                    target_entropy_loss = target_entropy_loss.mean()
                    
                    lagrangian_loss = -lagrangian * jax.lax.stop_gradient(kl_loss - cfg.kl_bound)
                    lagrangian_loss = lagrangian_loss.mean()
                    
                    loss = jnp.mean(actor_loss)
                    if cfg.update_entropy_lagrangian:
                        loss += target_entropy_loss

                    if cfg.update_kl_lagrangian:
                        loss += lagrangian_loss

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
                        m_step_lagrangian_loss=lagrangian_loss,
                        m_step_lagrangian=lagrangian,
                        entropy=jnp.mean(minibatch.log_weights),
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
                output, actor_grads = actor_grad_fn(train_state_mb.actor.params)
                actor_train_state = train_state_mb.actor.apply_gradients(actor_grads)
                actor_metrics = output[1]
                train_state_mb = train_state_mb.replace(
                    actor=actor_train_state,
                )
                if cfg.log_gnorm:
                    actor_gnorm = utils.tree_norm(actor_grads)
                    actor_metrics["actor_gnorm"] = actor_gnorm
                return train_state_mb, {
                     **actor_metrics,
                 }

            key, shuffle_key = jax.random.split(key)
            mini_batch_size = (cfg.num_steps * cfg.num_envs) // cfg.num_mini_batches
            indices = jax.random.permutation(shuffle_key, cfg.num_steps * cfg.num_envs)
            minibatch_idxs = jax.tree.map(
                lambda x: x.reshape(
                    (cfg.num_mini_batches, mini_batch_size, *x.shape[1:])
                ),
                indices,
            )

            train_state, metrics = jax.lax.scan(
                minibatch_update, train_state, (minibatch_idxs, jax.random.split(key, cfg.num_mini_batches))
            )
            metrics = jax.tree.map(lambda x: x.mean(0), metrics)
            return (idx + 1, train_state), metrics

        (_, train_state), update_metrics_actor = jax.lax.scan(
            f=update_actor,
            init=(1, train_state),
            xs=jax.random.split(key, cfg.num_epochs_actor),
        )

        update_metrics_actor = jax.tree.map(lambda x: x[-1], update_metrics_actor)

        return train_state, update_metrics_actor

    # Create Vmapped and JITted individual execution parts
    jit_collect_rollout = jax.jit(jax.vmap(collect_rollout))
    jit_compute_targets = jax.jit(jax.vmap(compute_targets))
    jit_train_critic = jax.jit(jax.vmap(train_critic))
    jit_update_Q = jax.jit(jax.vmap(do_update_Q))
    jit_train_actor = jax.jit(jax.vmap(train_actor))

    def train_fn(key: PRNGKeyArray, cfg: ReppoConfig) -> tuple[SACTrainState, dict]:
        eval_interval = int(
            (cfg.total_time_steps / (cfg.num_steps * cfg.num_envs)) // cfg.num_eval
        )
        num_train_steps = cfg.total_time_steps // (cfg.num_steps * cfg.num_envs)

        key, init_key = jax.random.split(key)
        train_state = jax.vmap(make_init(cfg, env, env_params))(
            jax.random.split(init_key, num_seeds)
        )

        times_accum = {
            "env_step": 0.0,
            "target_comp": 0.0,
            "critic_update": 0.0,
            "q_update": 0.0,
            "actor_update": 0.0
        }

        global_start_time = time.perf_counter()
        log_data = {}

        for i in range(num_train_steps):
            key, r_key, c_key, a_key = jax.random.split(key, 4)

            # 1. Env Stepping
            t0 = time.perf_counter()
            batch, train_state = jit_collect_rollout(jax.random.split(r_key, num_seeds), train_state)
            jax.tree.map(jax.block_until_ready, batch)
            t1 = time.perf_counter()

            # 2. Target Computation
            data = jit_compute_targets(batch)
            jax.tree.map(jax.block_until_ready, data)
            t2 = time.perf_counter()

            # 3. Critic Update
            train_state, metrics_critic = jit_train_critic(jax.random.split(c_key, num_seeds), train_state, data)
            jax.tree.map(jax.block_until_ready, train_state)
            t3 = time.perf_counter()

            # 4. Q Update
            data = jit_update_Q(train_state, data)
            jax.tree.map(jax.block_until_ready, data)
            t4 = time.perf_counter()

            # 5. Actor Update
            train_state, metrics_actor = jit_train_actor(jax.random.split(a_key, num_seeds), train_state, data)
            jax.tree.map(jax.block_until_ready, train_state)
            t5 = time.perf_counter()

            train_state = train_state.replace(iteration=train_state.iteration + 1)

            # Accumulate benchmark timings
            times_accum["env_step"] += (t1 - t0)
            times_accum["target_comp"] += (t2 - t1)
            times_accum["critic_update"] += (t3 - t2)
            times_accum["q_update"] += (t4 - t3)
            times_accum["actor_update"] += (t5 - t4)

            # Log periodically according to interval logic
            if (i + 1) % eval_interval == 0 or (i + 1) == num_train_steps:
                metrics = {**metrics_critic, **metrics_actor}

                # Average time components over the eval_interval length and shape to match (num_seeds,) dimensions
                metrics["time/env_step"] = jnp.array([times_accum["env_step"] / eval_interval] * num_seeds)
                metrics["time/target_comp"] = jnp.array([times_accum["target_comp"] / eval_interval] * num_seeds)
                metrics["time/critic_update"] = jnp.array([times_accum["critic_update"] / eval_interval] * num_seeds)
                metrics["time/q_update"] = jnp.array([times_accum["q_update"] / eval_interval] * num_seeds)
                metrics["time/actor_update"] = jnp.array([times_accum["actor_update"] / eval_interval] * num_seeds)

                abs_time = time.perf_counter() - global_start_time
                metrics["time/absolute"] = jnp.array([abs_time] * num_seeds)

                for k in times_accum: times_accum[k] = 0.0

                train_returns = {
                    "train/episode_return": train_state.last_env_state.info[
                        "returned_episode_returns"
                    ].mean(axis=-1), # Aggregate over num_envs
                    "train/episode_length": train_state.last_env_state.info[
                        "returned_episode_lengths"
                    ].mean(axis=-1),
                }

                log_data = {
                    "time_step": train_state.time_steps,
                    **utils.prefix_dict("train", metrics),
                    **train_returns,
                }

                if log_callback is not None:
                    log_callback(train_state, log_data)

        return train_state, log_data

    return train_fn

# type object
def run(cfg: DictConfig, trial: optuna.Trial | None) -> float:
    utils.setup_compilation_cache()
    sweep_metrics =[]

    if trial is not None:
        for name, values in cfg.trial_spec.items():
            if name in cfg.hyperparameters:
                sampled_value = _get_optuna_type(trial, name, values)
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

    metric_history =[]

    def log_callback(state, metrics):
        metrics["sys_time"] = time.perf_counter()
        if len(metric_history) > 0:
            # We track array of size (num_seeds,) so we access index 0
            num_env_steps = state.time_steps[0] - metric_history[-1]["time_step"][0]
            seconds = metrics["sys_time"] - metric_history[-1]["sys_time"]
            sps = num_env_steps / seconds
        else:
            sps = 0

        metric_history.append(metrics)
        episode_return = metrics["train/episode_return"].mean()
        train_length = metrics["train/episode_length"].mean()
        
        log_msg = f"step={state.time_steps[0]} train_return={episode_return:.3f}, train_length={train_length:.3f} sps={sps:.2f}"
        
        # Log breakdown timings
        log_msg += f" | env={metrics.get('train/time/env_step', [0])[0]*1000:.1f}ms target={metrics.get('train/time/target_comp', [0])[0]*1000:.1f}ms"
        log_msg += f" critic={metrics.get('train/time/critic_update', [0])[0]*1000:.1f}ms q={metrics.get('train/time/q_update', [0])[0]*1000:.1f}ms actor={metrics.get('train/time/actor_update', [0])[0]*1000:.1f}ms"

        logging.info(log_msg)
        
        log_data = {
            "train/episode_return": episode_return,
            "train/episode_length": train_length,
            "sps": sps,
            **jax.tree.map(jnp.mean, utils.filter_prefix("train", metrics)),
        }
        
        wandb.log(log_data, step=int(state.time_steps[0]))

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
    else:
        # The benchmark is the fully-jitted monolith: env.step is staged into one
        # lax.scan, which only works for JAX-native envs. Host-stepped envs
        # (maniskill, humanoid_bench) live in src/jaxrl/amdp_reppo.py.
        raise ValueError(
            f"The benchmark trainer supports only JAX-native envs (mjx); got "
            f"env.type={cfg.env.type!r}. Use src.jaxrl.amdp_reppo for host-stepped envs."
        )

    train_fn = make_train_fn(
        cfg=ReppoConfig(**cfg.hyperparameters),
        env=env,
        eval_env=eval_env,
        log_callback=log_callback,
        num_seeds=cfg.num_seeds,
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
        
        # train_fn operates sequentially internally to benchmark components, not JITted globally
        _, metrics = train_fn(key, ReppoConfig(**cfg.hyperparameters))
        
        duration = time.perf_counter() - start

        logging.info(f"Training took {duration:.2f} seconds.")
        # jnp.savez("metrics.npz", **metrics)
        wandb.finish()

        sweep_metrics.append(metrics["train/episode_return"])

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
