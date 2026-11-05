"""TD(lambda) target recursion (src.jaxrl.utils.compute_lambda_targets).

The reference below re-derives the targets with an explicit O(T^2) forward expansion
instead of the reverse scan, so it exercises the same definition by a different route.

Semantics as implemented (see the function's docstring):
  * `value[t]`      = V of the bootstrap state for step t.
  * `truncated[t]`  = step t ended with a value bootstrap.
  * `done[t]`       = step t ended the episode with a zeroed bootstrap.
  * carry mode (default, JAX-native envs): the truncation flag is carried through the
    reverse scan, seeded with 1 so the last rollout step bootstraps its tail.
  * own-step mode (`bootstrap_at_own_step=True`, host envs): the flag applies to its
    own step, replicating the torch/upstream-REPPO `compute_gve` loop; a NumPy port of
    that loop is the second reference below.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from src.jaxrl.types import Transition
from src.jaxrl.utils import compute_lambda_targets


def _make_batch(reward, value, done, truncated, importance_weight):
    """Wrap raw arrays in a Transition; unused fields get zeros of the right shape."""
    steps, envs = reward.shape
    zeros = jnp.zeros((steps, envs))
    return Transition(
        obs=jnp.zeros((steps, envs, 1)),
        critic_obs=jnp.zeros((steps, envs, 1)),
        prior_action=jnp.zeros((steps, envs, 1)),
        action=jnp.zeros((steps, envs, 1)),
        action_unsquashed=jnp.zeros((steps, envs, 1)),
        squashing_correction_grad=jnp.zeros((steps, envs, 1)),
        squashing_correction_val=zeros,
        reward=reward,
        soft_reward=reward,
        next_emb=jnp.zeros((steps, envs, 1)),
        value=value,
        done=done,
        truncated=truncated,
        importance_weight=importance_weight,
        log_weights=jnp.zeros((steps, envs, 1)),
        log_path_weight_deterministic=jnp.zeros((steps, envs, 1)),
        log_path_weight_stochastic=jnp.zeros((steps, envs, 1)),
        log_p_T_ref=jnp.zeros((steps, envs, 1)),
        Q_value=zeros,
        Q_score=jnp.zeros((steps, envs, 1)),
        info={},
    )


def _reference_targets(reward, value, done, truncated, importance_weight, gamma, lmbda):
    """Slow per-env reference: expand the recursion explicitly, latest step first."""
    steps, envs = reward.shape
    targets = np.zeros((steps, envs))
    for e in range(envs):
        # carry as seeded by the implementation at the rollout tail
        lambda_return = float(value[steps - 1, e])
        carry_truncated = 1.0
        carry_iw = 0.0
        for t in range(steps - 1, -1, -1):
            c = np.exp(carry_iw) * lmbda
            lambda_sum = c * lambda_return + (1.0 - c) * float(value[t, e])
            if carry_truncated:
                bootstrap = float(value[t, e])
            else:
                bootstrap = (1.0 - float(done[t, e])) * lambda_sum
            lambda_return = float(reward[t, e]) + gamma * bootstrap
            targets[t, e] = lambda_return
            carry_truncated = float(truncated[t, e])
            carry_iw = float(importance_weight[t, e])
    return targets


def _check(reward, value, done, truncated, importance_weight, gamma=0.99, lmbda=0.95):
    batch = _make_batch(reward, value, done, truncated, importance_weight)
    got = np.asarray(compute_lambda_targets(batch, gamma, lmbda))
    want = _reference_targets(
        np.asarray(reward),
        np.asarray(value),
        np.asarray(done),
        np.asarray(truncated),
        np.asarray(importance_weight),
        gamma,
        lmbda,
    )
    np.testing.assert_allclose(got, want, rtol=1e-5, atol=1e-5)
    return got


def test_matches_reference_on_random_rollouts():
    """Random rewards/values/dones: scan implementation == explicit expansion."""
    key = jax.random.PRNGKey(0)
    k1, k2, k3, k4 = jax.random.split(key, 4)
    steps, envs = 16, 8
    reward = jax.random.normal(k1, (steps, envs))
    value = jax.random.normal(k2, (steps, envs))
    done = (jax.random.uniform(k3, (steps, envs)) < 0.2).astype(jnp.float32)
    # truncation only occurs where the episode ended
    truncated = done * (jax.random.uniform(k4, (steps, envs)) < 0.5).astype(jnp.float32)
    _check(reward, value, done, truncated, jnp.zeros((steps, envs)))


def test_no_done_is_geometric_lambda_mixture():
    """With no episode boundary and lmbda=0, the target is a one-step soft TD backup."""
    steps, envs = 5, 1
    reward = jnp.ones((steps, envs))
    value = jnp.full((steps, envs), 2.0)
    zeros = jnp.zeros((steps, envs))
    gamma = 0.9
    got = _check(reward, value, zeros, zeros, zeros, gamma=gamma, lmbda=0.0)
    # lmbda=0: bootstrap collapses to V(s_{t+1}) for every non-tail step
    expected_interior = 1.0 + gamma * 2.0
    np.testing.assert_allclose(got[:-1, 0], expected_interior, rtol=1e-6)


def test_termination_zeroes_the_bootstrap():
    """A terminal step (done, not truncated) drops the bootstrap: target == reward."""
    steps, envs = 4, 1
    reward = jnp.arange(1.0, steps + 1).reshape(steps, envs)
    value = jnp.full((steps, envs), 10.0)
    done = jnp.array([[0.0], [1.0], [0.0], [0.0]])
    truncated = jnp.zeros((steps, envs))
    got = _check(reward, value, done, truncated, jnp.zeros((steps, envs)))
    # step 1 terminates and its successor is not truncated -> pure reward
    np.testing.assert_allclose(got[1, 0], reward[1, 0], rtol=1e-6)


def test_carried_truncation_forces_value_bootstrap():
    """Where the carried truncation flag is set, the target bootstraps V."""
    steps, envs = 4, 1
    reward = jnp.ones((steps, envs))
    value = jnp.full((steps, envs), 5.0)
    done = jnp.array([[0.0], [1.0], [0.0], [0.0]])
    truncated = jnp.array([[0.0], [1.0], [0.0], [0.0]])  # step 1 hit the time limit
    gamma = 0.9
    got = _check(reward, value, done, truncated, jnp.zeros((steps, envs)), gamma=gamma)
    # step 0 carries truncated_1 == 1 -> bootstraps with V(s_1) regardless of lmbda
    np.testing.assert_allclose(got[0, 0], 1.0 + gamma * 5.0, rtol=1e-6)


def test_tail_bootstraps_with_the_final_value():
    """The carry is seeded truncated=1, so the last step always bootstraps V(s_T)."""
    steps, envs = 3, 2
    reward = jnp.zeros((steps, envs))
    value = jnp.arange(steps * envs, dtype=jnp.float32).reshape(steps, envs)
    zeros = jnp.zeros((steps, envs))
    gamma = 0.5
    got = _check(reward, value, zeros, zeros, zeros, gamma=gamma)
    np.testing.assert_allclose(got[-1], gamma * np.asarray(value[-1]), rtol=1e-6)


def test_all_done_every_step():
    """Every step terminal: no step bootstraps except via the seeded tail flag."""
    steps, envs = 4, 3
    key = jax.random.PRNGKey(1)
    reward = jax.random.normal(key, (steps, envs))
    value = jnp.full((steps, envs), 3.0)
    done = jnp.ones((steps, envs))
    truncated = jnp.zeros((steps, envs))
    got = _check(reward, value, done, truncated, jnp.zeros((steps, envs)))
    # interior steps: done kills the bootstrap, carried truncation is 0 -> reward only
    np.testing.assert_allclose(got[:-1], np.asarray(reward)[:-1], rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("lmbda", [0.0, 0.5, 1.0])
def test_lambda_sweep_matches_reference(lmbda):
    """The mixture rate is honored across the full lambda range."""
    key = jax.random.PRNGKey(2)
    k1, k2 = jax.random.split(key)
    steps, envs = 10, 4
    reward = jax.random.normal(k1, (steps, envs))
    value = jax.random.normal(k2, (steps, envs))
    zeros = jnp.zeros((steps, envs))
    _check(reward, value, zeros, zeros, zeros, lmbda=lmbda)


def test_jit_matches_eager():
    """The recursion is jit-safe: same inputs, identical outputs."""
    key = jax.random.PRNGKey(3)
    k1, k2 = jax.random.split(key)
    steps, envs = 8, 2
    reward = jax.random.normal(k1, (steps, envs))
    value = jax.random.normal(k2, (steps, envs))
    zeros = jnp.zeros((steps, envs))
    batch = _make_batch(reward, value, zeros, zeros, zeros)
    eager = compute_lambda_targets(batch, 0.99, 0.95)
    jitted = jax.jit(compute_lambda_targets, static_argnums=(1, 2))(batch, 0.99, 0.95)
    np.testing.assert_array_equal(np.asarray(eager), np.asarray(jitted))


# --- own-step mode (host envs, torch/upstream-REPPO compute_gve semantics) ---


def _torch_gve_reference(reward, value, done, truncated, gamma, lmbda):
    """Direct NumPy port of the torch trainer's compute_gve loop."""
    steps, envs = reward.shape
    truncated = np.asarray(truncated, dtype=np.float64).copy()
    truncated[-1] = 1.0
    gves = np.zeros((steps, envs))
    last_gve = np.zeros(envs)
    for t in range(steps - 1, -1, -1):
        lambda_sum = lmbda * last_gve + (1.0 - lmbda) * value[t]
        delta = gamma * np.where(
            truncated[t] > 0, value[t], (1.0 - done[t]) * lambda_sum
        )
        last_gve = reward[t] + delta
        gves[t] = last_gve
    return gves


def _check_own_step(reward, value, done, truncated, gamma=0.99, lmbda=0.95):
    zeros_iw = jnp.zeros(reward.shape)
    batch = _make_batch(reward, value, done, truncated, zeros_iw)
    got = np.asarray(
        compute_lambda_targets(batch, gamma, lmbda, bootstrap_at_own_step=True)
    )
    want = _torch_gve_reference(
        np.asarray(reward),
        np.asarray(value),
        np.asarray(done),
        np.asarray(truncated),
        gamma,
        lmbda,
    )
    np.testing.assert_allclose(got, want, rtol=1e-5, atol=1e-5)
    return got


def test_own_step_matches_torch_gve_partial_reset_flags():
    """ManiSkill flag convention: done == 0 everywhere, ends flagged as truncations."""
    key = jax.random.PRNGKey(4)
    k1, k2, k3 = jax.random.split(key, 3)
    steps, envs = 16, 8
    reward = jax.random.normal(k1, (steps, envs))
    value = jax.random.normal(k2, (steps, envs))
    done = jnp.zeros((steps, envs))
    truncated = (jax.random.uniform(k3, (steps, envs)) < 0.2).astype(jnp.float32)
    _check_own_step(reward, value, done, truncated)


def test_own_step_matches_torch_gve_hb_flags():
    """HB flag convention: done = termination|truncation, truncated = time-outs only."""
    key = jax.random.PRNGKey(5)
    k1, k2, k3, k4 = jax.random.split(key, 4)
    steps, envs = 16, 8
    reward = jax.random.normal(k1, (steps, envs))
    value = jax.random.normal(k2, (steps, envs))
    done = (jax.random.uniform(k3, (steps, envs)) < 0.2).astype(jnp.float32)
    truncated = done * (jax.random.uniform(k4, (steps, envs)) < 0.5).astype(jnp.float32)
    _check_own_step(reward, value, done, truncated)


def test_own_step_truncation_bootstraps_its_own_value():
    """The truncated step's target is r + gamma*V(bootstrap state) itself, and the step
    before it lambda-mixes through the boundary (torch semantics, no early cut)."""
    steps, envs = 4, 1
    reward = jnp.ones((steps, envs))
    value = jnp.full((steps, envs), 5.0)
    done = jnp.zeros((steps, envs))
    truncated = jnp.array([[0.0], [1.0], [0.0], [0.0]])
    gamma, lmbda = 0.9, 0.95
    got = _check_own_step(reward, value, done, truncated, gamma=gamma, lmbda=lmbda)
    target_1 = 1.0 + gamma * 5.0
    np.testing.assert_allclose(got[1, 0], target_1, rtol=1e-6)
    lambda_sum_0 = lmbda * target_1 + (1.0 - lmbda) * 5.0
    np.testing.assert_allclose(got[0, 0], 1.0 + gamma * lambda_sum_0, rtol=1e-6)


def test_own_step_tail_is_forced_truncated():
    """The last rollout step bootstraps its value even if flagged done (torch quirk:
    truncated[-1] = 1 wins over done in the where)."""
    steps, envs = 3, 1
    reward = jnp.zeros((steps, envs))
    value = jnp.full((steps, envs), 7.0)
    done = jnp.array([[0.0], [0.0], [1.0]])
    truncated = jnp.zeros((steps, envs))
    gamma = 0.5
    got = _check_own_step(reward, value, done, truncated, gamma=gamma)
    np.testing.assert_allclose(got[-1, 0], gamma * 7.0, rtol=1e-6)
