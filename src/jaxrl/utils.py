import os

import jax
import jax.numpy as jnp
import optuna
from jaxtyping import Array, Float, PyTree

from src.jaxrl.types import Transition


def setup_compilation_cache():
    """Enable the persistent XLA compilation cache so repeated runs skip recompilation."""
    jax.config.update(
        "jax_compilation_cache_dir", os.path.expanduser("~/.cache/reppo_jax_cache")
    )
    jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 1.0)


def prefix_dict(prefix: str, metrics: dict, sep: str = "/") -> dict:
    """Add a prefix to all keys in a dictionary."""
    return {f"{prefix}{sep}{key}": value for key, value in metrics.items()}


def filter_prefix(prefix: str, metrics: dict, sep: str = "/") -> dict:
    """Filter keys in a dictionary by a prefix."""
    return {
        key: value for key, value in metrics.items() if key.startswith(prefix + sep)
    }


def hl_gauss(
    inp: Float[Array, "*batch 1"],
    num_bins: int,
    vmin: float,
    vmax: float,
    epsilon: float = 0.0,
) -> Float[Array, "*batch num_bins"]:
    """Converts a batch of scalars to soft two-hot encoded targets for discrete regression."""
    x = jnp.clip(inp, vmin, max=vmax).squeeze() / (1 - epsilon)
    bin_width = (vmax - vmin) / (num_bins - 1)
    sigma_to_final_sigma_ratio = 0.75
    support = jnp.linspace(
        vmin - bin_width / 2, vmax + bin_width / 2, num_bins + 1, dtype=jnp.float32
    )
    sigma = bin_width * sigma_to_final_sigma_ratio
    cdf_evals = jax.scipy.special.erf((support - x) / (jnp.sqrt(2) * sigma))
    z = cdf_evals[-1] - cdf_evals[0]
    target_probs = cdf_evals[1:] - cdf_evals[:-1]
    target_probs = (target_probs / z).reshape(*inp.shape[:-1], num_bins)

    uniform = jnp.ones_like(target_probs) / num_bins

    return (1 - epsilon) * target_probs + epsilon * uniform

def multi_softmax(
    x: Float[Array, "*batch d"], dim: int | None = 8, get_logits: bool = False
) -> Float[Array, "*batch d"]:
    inp_shape = x.shape
    if dim is not None:
        x = x.reshape(*x.shape[:-1], -1, dim)
    if get_logits:
        x = jax.nn.log_softmax(x, axis=-1)
    else:
        x = jax.nn.softmax(x, axis=-1)
    return x.reshape(*inp_shape)


def tree_norm(tree: PyTree[Float[Array, "..."]]) -> Float[Array, ""]:
    return jnp.sqrt(sum((x**2).sum() for x in jax.tree_util.tree_leaves(tree)))

def count_params(params: PyTree[Float[Array, "..."]]) -> int:
    return sum([p.size for p in jax.tree_util.tree_leaves(params)])


def compute_reverse_ess(log_weights: Float[Array, "n"]) -> Float[Array, ""]:
    # Subtract the maximum log weight for numerical stability
    max_log_weight = jnp.max(log_weights)
    stable_log_weights = log_weights - max_log_weight

    # Compute the importance weights in a numerically stable way
    is_weights = jnp.exp(stable_log_weights)

    # Compute the sums needed for ESS
    sum_is_weights = jnp.sum(is_weights)
    sum_is_weights_squared = jnp.sum(is_weights ** 2)

    # Calculate the effective sample size (ESS)
    ess = (sum_is_weights ** 2) / (log_weights.shape[0] * sum_is_weights_squared)

    return ess


def staggered_reset_offsets(
    num_envs: int,
    max_episode_steps: int,
    num_steps: int,
    num_stagger_blocks: int | None = None,
) -> Float[Array, "envs"]:
    """Deterministic per-env elapsed-step offsets that desynchronize episode ends.

    Port of the torch trainer's `create_staggered_resets`: envs are split into
    contiguous groups and group g starts `g * (max_episode_steps // num_groups)` steps
    into its episode. The group count defaults to `max_episode_steps // num_steps`, so
    roughly one group truncates per rollout.
    """
    if num_stagger_blocks is not None and num_stagger_blocks > 0:
        num_groups = num_stagger_blocks
    else:
        num_groups = max(1, max_episode_steps // num_steps) if num_steps > 0 else 1
    num_groups = max(1, min(num_groups, num_envs))
    envs_per_group = max(1, num_envs // num_groups)
    group_idx = jnp.clip(jnp.arange(num_envs) // envs_per_group, 0, num_groups - 1)
    return (group_idx * (max_episode_steps // num_groups)).astype(jnp.float32)


def compute_lambda_targets(
    batch: Transition,
    gamma: float,
    lmbda: float,
    bootstrap_at_own_step: bool = False,
) -> Float[Array, "steps envs"]:
    """TD(lambda) targets on the soft rewards, computed by a reverse scan.

    `batch.value[t]` is the critic value of the bootstrap state for step t (V(s_{t+1}),
    computed from the pre-reset terminal obs where a host env auto-reset and surfaced
    it) and `batch.truncated[t]` flags a step whose target should bootstrap with that
    value. Each target mixes the next target with `value` at rate
    `exp(importance_weight) * lmbda` (the importance-weight factor is vestigial: the
    trainers always store zeros) and zeroes the bootstrap on `done`.

    The truncation flag is applied in one of two ways, matching the two autoreset
    conventions in this repo:

    * `bootstrap_at_own_step=False` — JAX-native envs (brax-style same-step autoreset).
      The terminal obs is unobtainable there: the obs at an episode end is already the
      reset obs, so the boundary step's own target cannot be formed. The flag is
      therefore carried one step through the reverse scan: the boundary step falls into
      the `done` branch (its target is excluded from the critic loss by the
      `1 - truncated` mask) and the step before it bootstraps V(s_t) directly instead
      of lambda-mixing into the excluded step. Same idea as brax PPO's truncation_mask.
    * `bootstrap_at_own_step=True` — host-stepped envs, replicating the torch/upstream
      REPPO `compute_gve`: the step flagged `truncated` bootstraps its own `value` and
      lambda-mixing runs through everywhere else. The last rollout step is treated as
      truncated so the tail bootstraps.

    In both modes the seeded/forced tail flag makes the last rollout step bootstrap
    with its `value`, regardless of `done`.
    """

    def compute_nstep_lambda(carry, xs):
        lambda_return, carried_truncated, importance_weight = carry
        transition, own_truncated = xs
        # combine importance_weights with TD lambda
        done = transition.done
        reward = transition.soft_reward
        value = transition.value
        truncated = own_truncated if bootstrap_at_own_step else carried_truncated
        lambda_sum = (
            jnp.exp(importance_weight) * lmbda * lambda_return
            + (1 - jnp.exp(importance_weight) * lmbda) * value
        )
        delta = gamma * jnp.where(truncated, value, (1.0 - done) * lambda_sum)
        lambda_return = reward + delta
        return (
            lambda_return,
            transition.truncated,
            transition.importance_weight,
        ), lambda_return

    if bootstrap_at_own_step:
        own_truncated = batch.truncated.at[-1].set(1.0)
    else:
        own_truncated = batch.truncated

    _, target_values = jax.lax.scan(
        compute_nstep_lambda,
        (
            batch.value[-1],
            jnp.ones_like(batch.truncated[0]),
            jnp.zeros_like(batch.importance_weight[0]),
        ),
        (batch, own_truncated),
        reverse=True,
    )
    return target_values


def _get_optuna_type(trial: optuna.Trial, name, values: list):
    if all(isinstance(v, int) for v in values):
        return trial.suggest_int(name, low=min(values), high=max(values))
    elif all(isinstance(v, float) for v in values):
        return trial.suggest_float(name, low=min(values), high=max(values))
    elif all(isinstance(v, str) for v in values):
        return trial.suggest_categorical(name, values)
    elif all(isinstance(v, bool) for v in values):
        return trial.suggest_categorical(name, [True, False])
    else:
        raise ValueError("Values must be of the same type (int, float, or str).")
