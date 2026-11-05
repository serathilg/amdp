from typing import Any

import jax
from flax import nnx, struct
from gymnax.environments.environment import EnvState
from jaxtyping import Array, Float


class Transition(struct.PyTreeNode):
    """Rollout batch in the canonical "steps envs ..." layout.

    Shapes describe the stacked output of the rollout scan; the learn step
    later reshapes to flat minibatches ("(steps envs) ..."), where the same
    field names apply with the leading axes fused. Annotations are
    documentation, not runtime-checked.
    """

    obs: Float[Array, "steps envs obs"]
    critic_obs: Float[Array, "steps envs critic_obs"]
    prior_action: Float[Array, "steps envs act"]
    action: Float[Array, "steps envs act"]
    action_unsquashed: Float[Array, "steps envs act"]
    squashing_correction_grad: Float[Array, "steps envs act"]
    squashing_correction_val: Float[Array, "steps envs"]
    reward: Float[Array, "steps envs"]
    soft_reward: Float[Array, "steps envs"]
    next_emb: Float[Array, "steps envs emb"]
    value: Float[Array, "steps envs"]
    done: Float[Array, "steps envs"]
    truncated: Float[Array, "steps envs"]
    importance_weight: Float[Array, "steps envs"]
    log_weights: Float[Array, "steps envs 1"]
    log_path_weight_deterministic: Float[Array, "steps envs 1"]
    log_path_weight_stochastic: Float[Array, "steps envs 1"]
    log_p_T_ref: Float[Array, "steps envs 1"]
    Q_value: Float[Array, "steps envs"]
    Q_score: Float[Array, "steps envs act"]
    info: dict[str, jax.Array]


class ReppoConfigBase(struct.PyTreeNode):
    """Shared hyperparameter dataclass of the JAX trainers.

    Each trainer subclasses this as `ReppoConfig`, adding only its own
    evaluation field (`best_of_n_values` for the Brownian trainer, `eval_ode`
    for the OU trainer) so that
    `ReppoConfig(**cfg.hyperparameters)` keeps rejecting the other trainer's
    keys (hydra silently re-merges stale keys from archived run configs; this
    TypeError is the only guard).
    """

    lr: float
    gamma: float
    total_time_steps: int
    num_steps: int
    lmbda: float
    num_mini_batches: int
    num_envs: int
    num_subproc: int
    num_epochs: int
    num_epochs_actor: int
    num_epochs_critic: int
    batch_repetitions: int
    Q_score_max_norm: float
    max_grad_norm: float | None
    normalize_env: bool
    polyak: float
    exploration_noise_min: float
    exploration_noise_max: float
    exploration_base_envs: int
    ent_start: float
    ent_target_mult: float
    kl_start: float
    num_eval: int = 25
    max_episode_steps: int = 1000
    critic_hidden_dim: int = 512
    vmin: int = -100
    vmax: int = 100
    num_bins: int = 250
    hl_gauss: bool = False
    kl_bound: float = 1.0
    aux_loss_mult: float = 0.0
    update_kl_lagrangian: bool = True
    update_entropy_lagrangian: bool = True
    use_critic_norm: bool = True
    num_critic_encoder_layers: int = 1
    num_critic_head_layers: int = 1
    num_critic_pred_layers: int = 1
    use_simplical_embedding: bool = False
    use_critic_skip: bool = False
    reduce_kl: bool = True
    anneal_lr: bool = False
    # percentile on jnp.percentile's [0, 100] scale
    Q_score_max_percentile: float = 95.0
    Q_score_max_norm_for_squashed: bool = False
    use_target_critic_for_actor: bool = False
    eval_first_episode_only: bool = False
    # Initial elapsed steps: deterministic group stagger (torch parity) if True, the
    # original JAX random per-env offsets if False. Either way they keep the envs from
    # stepping in tandem; no effect for humanoid_bench, whose subprocess TimeLimit
    # cannot be pre-aged.
    use_staggered_resets: bool = False
    num_stagger_blocks: int | None = None

    trust_region_lagrangian: str = "dual_descent"
    trust_region_time_weighting: bool = False
    trust_region_granularity: str = "avg"

    # logging settings
    log_pnorm: bool = False
    log_gnorm: bool = False

    # diffusion settings
    diffusion: Any = None # DictConfig


class SACTrainState(struct.PyTreeNode):
    critic: nnx.TrainState
    critic_target: nnx.TrainState
    actor: nnx.TrainState
    actor_target: nnx.TrainState
    iteration: int
    time_steps: int
    last_env_state: EnvState
    last_obs: jax.Array
    last_critic_obs: jax.Array
